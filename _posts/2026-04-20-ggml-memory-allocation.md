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
    └── Sets tensor->data pointers into the GPU buffer
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