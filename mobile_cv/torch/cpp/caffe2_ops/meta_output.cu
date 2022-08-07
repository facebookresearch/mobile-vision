#include "meta_output.h"

#include "caffe2/caffe2/core/context_gpu.h"

namespace caffe2 {

REGISTER_CUDA_OPERATOR(MetaOutput, MetaOutputOp<CUDAContext>);

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CUDA(
    MetaOutput,
    caffe2::MetaOutputOp<caffe2::CUDAContext>);
