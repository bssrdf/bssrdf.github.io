# Understanding Swizzling in cutlass using implicit GEMM convolution example


The cutlass implicit GEMM convolution example is in [here](https://github.com/NVIDIA/cutlass/blob/main/examples/16_ampere_tensorop_conv2dfprop/ampere_tensorop_conv2dfprop.cu)


The ```DefaultConv2dFprop``` is [here](https://github.com/bssrdf/cutlass/blob/5b76420d6ae0ec0dbf82dc19317890551bffb1a6/include/cutlass/conv/kernel/default_conv2d_fprop.h#L1061)

The ```DefaultMmaCore``` specialization is [here](https://github.com/bssrdf/cutlass/blob/5b76420d6ae0ec0dbf82dc19317890551bffb1a6/include/cutlass/gemm/threadblock/default_mma_core_sm80.h#L1390)


## Shared memory swizzling implemented by cutlass

The implicit GEMM example uses ```cp.async``` to load activation and filter tensors from
global memory to shared memory. The code does this is [here](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/conv/threadblock/implicit_gemm_multistage.h) with relevant lines as below
```
void copy_tiles_and_advance(
    IteratorA &iterator_A, IteratorB &iterator_B,
    int group_start_A = 0, int group_start_B = 0) {

    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);
      
    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {

      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                  dst_ptr + v, iterator_A.get(), iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);

    this->smem_iterator_B_.set_iteration_index(group_start_B);
    
    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());
        
        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                  dst_ptr + v, iterator_B.get(), iterator_B.valid());

          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
}
```
Here ```this->smem_iterator_A_.get()``` returns a pointer to the location in shared memory where the element is being accessed. Note that the address is already pointing to permuted/swizzled offset. The key class for achieving this is ```RegularTileAccessIterator``` specialized with ```layout::TensorOpMultiplicandCrosswise<                               sizeof_bits<Element_>::value, Crosswise>``` located [here](https://github.com/NVIDIA/cutlass/blob/5b76420d6ae0ec0dbf82dc19317890551bffb1a6/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h#L435). Here ```TensorRef``` is templated as ```TensorRef<Element, Layout>``` where ```Layout``` is of type ```template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCrosswise```. In this struct, the address swizzling is done by a [Base object](https://github.com/NVIDIA/cutlass/blob/5b76420d6ae0ec0dbf82dc19317890551bffb1a6/include/cutlass/layout/tensor_op_multiplicand_sm75.h#L151). The logic of swizzling is wrapped in ```operator()``` function of the Base Object, which is called in ```TensorRef<Element, Layout>```'s ```offset()``` function
```
/// Computes the offset of an index from the origin of the tensor
  CUTLASS_HOST_DEVICE
  LongIndex offset(TensorCoord const& coord) const {
    return layout_(coord); // layout is template <int ElementSize, int Crosswise>
                          //            struct RowMajorTensorOpMultiplicandCrosswise
  }
```
In ```RegularTileAccessIterator```'s constructor, ```TensorRef<Element, Layout>```'s ```offset()``` function is called:
```
CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : sections_(ref.stride(0) / kCrosswise),
        sections_per_stage_(Shape::kContiguous / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};
      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    ref.offset(thread_offset_in_threadblock_tile) /
                        Layout::kElementsPerAccess;
    }

    set_iteration_index(0);
  }
```













| ROW | T0 | T1 | T2 | T3
| --- | --- | --- | --- | ---
| ROW 0 | 7.0,2.0,8.0,8.0,2.0,2.0,5.0,7.0 | 8.0,7.0,6.0,2.0,1.0,1.0,1.0,8.0 | 8.0,8.0,6.0,4.0,3.0,9.0,2.0,0.0 | 4.0,7.0,7.0,5.0,8.0,4.0,7.0,6.0
| ROW 1 | 7.0,0.0,1.0,7.0,6.0,0.0,4.0,4.0 | 0.0,5.0,8.0,3.0,4.0,3.0,2.0,0.0 | 7.0,3.0,3.0,9.0,9.0,7.0,9.0,8.0 | 5.0,0.0,1.0,9.0,8.0,1.0,1.0,8.0
| ROW 2 | 8.0,5.0,0.0,4.0,9.0,2.0,7.0,9.0 | 4.0,5.0,8.0,1.0,4.0,7.0,0.0,0.0 | 5.0,4.0,2.0,2.0,1.0,9.0,3.0,6.0 | 8.0,1.0,9.0,3.0,2.0,9.0,8.0,1.0
| ROW 3 | 4.0,5.0,0.0,6.0,1.0,0.0,1.0,5.0 | 6.0,6.0,4.0,7.0,8.0,2.0,6.0,9.0 | 8.0,1.0,2.0,4.0,6.0,9.0,7.0,5.0 | 1.0,6.0,1.0,7.0,8.0,1.0,8.0,7.0
| ROW 4 | 8.0,2.0,5.0,7.0,9.0,5.0,2.0,2.0 | 7.0,9.0,0.0,3.0,0.0,8.0,7.0,2.0 | 0.0,4.0,2.0,6.0,6.0,6.0,2.0,6.0 | 8.0,5.0,8.0,2.0,4.0,5.0,0.0,5.0
| ROW 5 | 3.0,2.0,7.0,5.0,8.0,5.0,4.0,0.0 | 9.0,0.0,1.0,4.0,7.0,6.0,5.0,4.0 | 1.0,6.0,3.0,5.0,1.0,0.0,4.0,8.0 | 0.0,5.0,6.0,4.0,4.0,3.0,4.0,3.0
| ROW 6 | 3.0,8.0,1.0,1.0,3.0,9.0,0.0,6.0 | 1.0,4.0,0.0,5.0,4.0,5.0,8.0,7.0 | 0.0,3.0,3.0,4.0,9.0,9.0,6.0,0.0 | 0.0,6.0,0.0,3.0,8.0,1.0,2.0,5.0
| ROW 7 | 3.0,0.0,0.0,3.0,7.0,0.0,9.0,3.0 | 2.0,8.0,7.0,2.0,3.0,4.0,0.0,1.0 | 5.0,4.0,6.0,0.0,3.0,8.0,0.0,6.0 | 3.0,1.0,6.0,3.0,3.0,2.0,1.0,7.0
| ROW 8 | 3.0,5.0,0.0,2.0,7.0,6.0,1.0,1.0 | 4.0,4.0,1.0,7.0,9.0,4.0,7.0,7.0 | 9.0,5.0,7.0,6.0,1.0,1.0,2.0,2.0 | 4.0,6.0,7.0,3.0,1.0,6.0,4.0,4.0
| ROW 9 | 6.0,9.0,8.0,9.0,9.0,6.0,8.0,7.0 | 5.0,1.0,1.0,1.0,3.0,7.0,9.0,1.0 | 1.0,2.0,0.0,9.0,3.0,3.0,3.0,9.0 | 2.0,6.0,9.0,1.0,0.0,1.0,0.0,1.0
| ROW 10 | 1.0,6.0,0.0,3.0,7.0,3.0,7.0,4.0 | 6.0,8.0,9.0,3.0,2.0,0.0,9.0,7.0 | 9.0,4.0,3.0,1.0,0.0,1.0,5.0,8.0 | 5.0,6.0,7.0,8.0,3.0,1.0,9.0,0.0
| ROW 11 | 7.0,7.0,5.0,5.0,3.0,6.0,7.0,2.0 | 5.0,5.0,6.0,3.0,8.0,6.0,5.0,4.0 | 8.0,3.0,5.0,1.0,0.0,5.0,5.0,7.0 | 3.0,8.0,7.0,6.0,0.0,1.0,9.0,2.0
| ROW 12 | 5.0,4.0,4.0,4.0,6.0,5.0,4.0,9.0 | 6.0,2.0,7.0,0.0,1.0,0.0,8.0,2.0 | 4.0,9.0,9.0,0.0,5.0,7.0,7.0,9.0 | 7.0,3.0,2.0,2.0,7.0,4.0,3.0,3.0
| ROW 13 | 6.0,9.0,8.0,1.0,1.0,7.0,2.0,8.0 | 5.0,2.0,8.0,7.0,9.0,7.0,9.0,0.0 | 5.0,2.0,4.0,7.0,5.0,3.0,7.0,1.0 | 9.0,0.0,0.0,7.0,9.0,0.0,7.0,9.0
| ROW 14 | 6.0,5.0,6.0,4.0,1.0,8.0,6.0,4.0 | 4.0,3.0,6.0,0.0,3.0,6.0,4.0,1.0 | 7.0,1.0,9.0,3.0,8.0,9.0,5.0,5.0 | 5.0,7.0,3.0,1.0,0.0,9.0,8.0,7.0
| ROW 15 | 7.0,0.0,9.0,4.0,0.0,3.0,0.0,3.0 | 5.0,5.0,3.0,0.0,5.0,4.0,5.0,1.0 | 6.0,6.0,3.0,9.0,8.0,0.0,5.0,1.0 | 8.0,8.0,7.0,4.0,5.0,0.0,9.0,5.0
