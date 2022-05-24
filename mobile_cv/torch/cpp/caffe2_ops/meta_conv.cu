#include "meta_conv.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(MetaConv, MetaConvOp<CUDAContext>);

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(
    MetaConv,
    caffe2::MetaConvOp<caffe2::CUDAContext>);
