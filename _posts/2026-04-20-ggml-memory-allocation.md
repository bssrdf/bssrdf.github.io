# How GGML allocates memory on Cuda backend

## The Flow

### 1. Backend Selection (lines ~200-218)

```cpp
#ifdef GGML_USE_CUDA
    if (n_gpu_layers > 0) {
        model.backend = ggml_backend_cuda_init(0);  // CUDA backend
    }
#endif
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();     // fallback
    }
```

Since `GGML_USE_CUDA` is defined and assuming `n_gpu_layers > 0`, `model.backend` points to the **CUDA backend**.

### 2. Weights Allocated on GPU (line ~278)

```cpp
model.buffer_w = ggml_backend_alloc_ctx_tensors(ctx, model.backend);
```

All model weights are allocated in **device (GPU) memory** via the CUDA backend.

### 3. Compute Buffer Allocation (lines ~820-830)

```cpp
allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
```

This is the critical line. `ggml_backend_get_default_buffer_type(model.backend)` returns the **CUDA buffer type** (since `model.backend` is the CUDA backend). So `allocr` is configured to allocate **temporary compute tensors in GPU memory**.

### 4. Worst-Case Graph for Pre-allocation (lines ~832-837)

```cpp
int n_tokens = std::min(model.hparams.n_ctx, params.n_batch);
int n_past = model.hparams.n_ctx - n_tokens;
struct ggml_cgraph * gf = gpt2_graph(model, n_past, n_tokens);

ggml_gallocr_reserve(allocr, gf);
size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);
```

`ggml_gallocr_reserve()` walks the graph, identifies all **intermediate tensors** (not the pre-allocated weights/KV), and allocates a **single contiguous buffer** in GPU memory large enough to hold all of them in the worst case.

### 5. Per-Invocation Allocation (inside `gpt2_eval`, lines ~756-758)

```cpp
struct ggml_cgraph * gf = gpt2_graph(model, n_past, embd_inp.size());

ggml_gallocr_alloc_graph(allocr, gf);
```

Each inference step creates a new graph (with different `N = embd_inp.size()`). `ggml_gallocr_alloc_graph()` **reuses the pre-allocated GPU buffer** and sets the `data` pointers on the intermediate tensors to point into that buffer. It does **not** allocate new memory — it just assigns pointers within the already-reserved GPU memory.

### 6. Execution (line ~780)

```cpp
ggml_backend_graph_compute(model.backend, gf);
```

The CUDA backend runs the graph. All intermediate tensors operate on **GPU memory** since their `data` pointers were set by `allocr` to point into the CUDA-allocated buffer.

## Summary Diagram

```
ggml_backend_get_default_buffer_type(model.backend)
    └── Returns: CUDA buffer type (device memory)
          │
          ▼
ggml_gallocr_new(cuda_buffer_type)
    └── Returns: allocr configured for GPU allocations
          │
          ▼
ggml_gallocr_reserve(allocr, worst_case_graph)
    └── Allocates ONE large GPU buffer (~compute buffer size)
          │
          ▼
Per token:
  ggml_gallocr_alloc_graph(allocr, gf)
    └── Sets tensor->data pointers into the GPU buffer(done by ggml_gallocr_init_tensor)
          │
          ▼
  ggml_backend_graph_compute(model.backend, gf)
    └── CUDA kernel executes, all tensors in GPU memory
```

## Key Insight

The `allocr` acts as a **per-graph temporary tensor allocator**. Since the backend is CUDA:

- **Weights** (`wte`, `wpe`, layer params) → allocated by `ggml_backend_alloc_ctx_tensors()` → **GPU memory**
- **KV memory** (`memory_k`, `memory_v`) → allocated by `ggml_backend_alloc_ctx_tensors()` → **GPU memory**
- **Intermediate compute tensors** (Q, K, V, KQ, KQ_scaled, KQV, etc.) → allocated by `ggml_gallocr` → **GPU memory** (from the pre-reserved buffer)

All three live in device memory, so no host-device transfers are needed during inference. The `ggml_gallocr` pattern avoids per-invocation `cudaMalloc`/`cudaFree` overhead by pre-allocating a large buffer and reusing it for temporary tensors across inference steps.

## Starting with ggml_gallocr graph allocator object

### Some low-level constructs
```
// relative memory address within an allocation that can be split into multiple buffers (chunks)
struct buffer_address {
    int chunk;     // index of a backend buffer
    size_t offset; // local memory offset within the buffer
};

static const struct buffer_address GGML_BUFFER_ADDRESS_INVALID = { -1, SIZE_MAX };

static bool ggml_buffer_address_less(struct buffer_address a, struct buffer_address b) {
    return a.chunk != b.chunk ? a.chunk < b.chunk : a.offset < b.offset;
}

struct free_block {
    size_t offset;
    size_t size;
};

struct tallocr_chunk {
    struct free_block free_blocks[MAX_FREE_BLOCKS];
    int n_free_blocks;
    size_t max_size;
};

struct ggml_dyn_tallocr {
    size_t alignment;
    size_t max_chunk_size;
    struct tallocr_chunk * chunks[GGML_VBUFFER_MAX_CHUNKS];
    int n_chunks;

};

struct hash_node {
    int n_children;
    int n_views;
    int buffer_id;
    struct buffer_address addr;
    bool allocated;
};

struct tensor_alloc {
    int buffer_id;
    struct buffer_address addr;
    size_t size_max; // 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
    struct tensor_alloc leaf;
};

struct node_alloc {
    struct tensor_alloc dst;
    struct tensor_alloc src[GGML_MAX_SRC];
};
```

Here's a summary of `ggml_gallocr_new_n` (lines 496–527 in ggml-alloc.c):

### Purpose

Creates a **graph allocator** (`ggml_gallocr_t`) that manages temporary tensor allocation for computation graphs across **multiple buffer types** (e.g., CUDA + CPU).

### What It Does

1. **Allocates the `ggml_gallocr` struct** and its internal arrays:
   - `bufts[]` — buffer type pointers (one per backend, e.g., CUDA buffer type)
   - `buffers[]` — virtual buffers (initially `NULL`, allocated later by `reserve`)
   - `buf_tallocs[]` — dynamic tensor allocators (one per buffer type)
   - Hash table for tensor → address mapping

2. **For each buffer type**, it checks if the same type appears multiple times and **shares the `ggml_dyn_tallocr`** to avoid duplicates.

3. **Creates a `ggml_dyn_tallocr`** per unique buffer type, initialized with:
   - **Alignment** — from `ggml_backend_buft_get_alignment(buft)`
   - **Max chunk size** — from `ggml_backend_buft_get_max_size(buft)`

### Key Detail

The `ggml_dyn_tallocr` is the **logical allocator** — it tracks free blocks and computes offsets, but does **not** allocate real GPU/CPU memory yet. The actual memory is allocated later when `ggml_gallocr_reserve()` is called, which triggers `ggml_vbuffer_alloc()` → `ggml_backend_buft_alloc_buffer()` → `cudaMalloc`.

### Signature

```c
ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
```

The simpler `ggml_gallocr_new(buft)` is just a wrapper that calls `ggml_gallocr_new_n(&buft, 1)`.


The actual assignment of memory address to tensors happens in `ggml-backend.cpp`

```
enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT((char *)addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}
```
which is called in `ggml-alloc.c`
```
static void ggml_vbuffer_tensor_alloc(struct vbuffer * buf, struct ggml_tensor * tensor, struct buffer_address buf_addr) {
    void * base = ggml_backend_buffer_get_base(buf->chunks[buf_addr.chunk]);
    void * addr = (char *)base + buf_addr.offset;
    ggml_backend_tensor_alloc(buf->chunks[buf_addr.chunk], tensor, addr);
}
```

which is called by `ggml_gallocr_init_tensor`.

