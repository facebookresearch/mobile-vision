#include "meta_conv.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MetaConv, MetaConvOp<CPUContext>);

OPERATOR_SCHEMA(MetaConv)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(
        R"DOC(Similar with AliasOp, storing the clipping limits of a conv op.
)DOC")
    .Arg("weight_l", "weight lower clipping limit")
    .Arg("weight_h", "weight upper clipping limit")
    .Arg("bias_l", "bias lower clipping limit")
    .Arg("bias_h", "bias upper clipping limit")
    .Arg("output_l", "output clipping limit")
    .Arg("output_h", "output upper clipping limit")
    .Input(0, "input", "Input tensor whose storage will be shared.")
    .Output(0, "output", "Tensor of same shape as input, sharing its storage.");

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    MetaConv,
    "_caffe2::MetaConv(Tensor input, float weight_l, float weight_h, float bias_l, float bias_h, float output_l, float output_h) -> (Tensor output)",
    caffe2::MetaConvOp<caffe2::CPUContext>);
