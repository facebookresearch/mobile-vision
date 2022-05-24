#include "meta_output.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(MetaOutput, MetaOutputOp<CPUContext>);

OPERATOR_SCHEMA(MetaOutput)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(Stores the output clipping limits of an op.)DOC")
    .Arg("output_l", "output clipping limit")
    .Arg("output_h", "output upper clipping limit")
    .Input(0, "input", "Input tensor whose storage will be shared.")
    .Output(0, "output", "Tensor of same shape as input, sharing its storage.");

} // namespace caffe2

C10_EXPORT_CAFFE2_OP_TO_C10_CPU(
    MetaOutput,
    "_caffe2::MetaOutput(Tensor input, float output_l, float output_h) -> (Tensor output)",
    caffe2::MetaOutputOp<caffe2::CPUContext>);
