# A Trick for Debugging GGML CUDA Backend Code

When debugging the CUDA backend of [GGML](https://github.com/ggml-org/llama.cpp/tree/master/ggml)
, it is important to understand CUDA’s asynchronous execution model. If you ignore this detail, you may end up inspecting the wrong tensor values and drawing incorrect conclusions.

A common debugging task is to inspect the contents of a tensor in a GGML computation graph (cgraph). The code that executes each tensor node typically looks like this:

```
bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
if (!ok) {
    GGML_LOG_ERROR("%s: op not supported %s (%s)\n",
        __func__, node->name, ggml_op_name(node->op));
}
GGML_ASSERT(ok);
```

A natural approach is to insert debugging code after the kernel launch to inspect the tensor values. For example:
```
bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);

printf(" %s, %s, %s, (%zu, %zu, %zu, %zu)\n", src->name,
       ggml_type_name(src->type), ggml_op_name(src->op),
       src->ne[0], src->ne[1], src->ne[2], src->ne[3]);

std::vector<float> data(ggml_nelements(src), 0.f);

ggml_backend_tensor_get(src, data.data(), 0, ggml_nbytes(src));

printf("[");
float vmin = 1.e16f, vmax = -1.e16f;
int nt = data.size();

for (int k = 0; k < nt; k++) {
    float val = data[k];

    if (isnan(val)) {
        ab = true;
        pos = k;
        break;
    }
    if (isinf(val)) {
        abi = true;
        pos = k;
        break;
    }

    vmin = min(vmin, val);
    vmax = max(vmax, val);
}

printf("] vmin,vmax %f, %f\n", vmin, vmax);
```

Here we use ggml_backend_tensor_get to copy tensor data from the CUDA backend into a std::vector, allowing us to inspect values or compute statistics such as min/max.

However, this code will often produce nonsense values. The reason lies in CUDA’s asynchronous execution behavior.

## Why the Tensor Values Look Wrong

There are two important details to consider:

- `ggml_backend_tensor_get` uses a different CUDA stream

Internally, `ggml_backend_tensor_get` performs a `cudaMemcpyAsync` to copy data from device to host. The copy is issued on CUDA stream 2 (`cudaStreamPerThread` in GGML). After issuing the copy, it calls:
```
cudaStreamSynchronize(cudaStreamPerThread);
```
to ensure the copy finishes before returning. This part works as expected.

- The kernel launches are also asynchronous

The problem originates from the call:
```
ggml_cuda_compute_forward(*cuda_ctx, node);
```
GGML launches CUDA kernels asynchronously on the default stream (stream 0), which can be accessed via:
```
cuda_ctx->stream()
```
Because kernel launches are asynchronous, when ggml_cuda_compute_forward returns, the computation may not have started or completed yet.

As a result, when `ggml_backend_tensor_get` copies the tensor data, the kernel that produces that tensor might still be pending. The buffer may therefore contain uninitialized or stale data.

## The Fix: Synchronize the Kernel Stream

To reliably inspect tensor values, you must synchronize the CUDA stream used for kernel execution before copying the tensor data.

Adding the following line solves the problem:
```
CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));
```
The corrected debugging code becomes:
```
bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);

printf(" %s, %s, %s, (%zu, %zu, %zu, %zu)\n", src->name,
       ggml_type_name(src->type), ggml_op_name(src->op),
       src->ne[0], src->ne[1], src->ne[2], src->ne[3]);

std::vector<float> data(ggml_nelements(src), 0.f);

CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

ggml_backend_tensor_get(src, data.data(), 0, ggml_nbytes(src));

// analyze tensor values...
```
Now the tensor data will reflect the actual results of the CUDA kernel, making debugging reliable.

## Summary

When debugging GGML CUDA kernels:

- Kernel launches are asynchronous.

- `ggml_backend_tensor_get` copies data on a different CUDA stream.

- Without synchronization, you may read invalid tensor data.

The solution is simple: synchronize the compute stream before reading tensors.
```
cudaStreamSynchronize(cuda_ctx->stream());
```
This small trick can save a lot of confusion when debugging CUDA code in GGML.

