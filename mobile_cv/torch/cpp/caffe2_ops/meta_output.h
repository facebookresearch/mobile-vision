#ifndef META_OUTPUT_OP_H_
#define META_OUTPUT_OP_H_

#include "caffe2/caffe2/core/context.h"
#include "caffe2/caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(MetaOutput)

namespace caffe2 {

template <class Context>
class MetaOutputOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit MetaOutputOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        output_l_(this->template GetSingleArgument<float>("output_l", 1.)),
        output_h_(this->template GetSingleArgument<float>("output_h", 1.)) {
    CAFFE_ENFORCE(OperatorBase::HasArgument("output_l"), "Requires output_l");
    CAFFE_ENFORCE(OperatorBase::HasArgument("output_h"), "Requires output_h");
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    CAFFE_ENFORCE_GE(input.numel(), 0, "Tensor is not initialized");

    // This doesn't work anymore as this is "newstyle" operator
    // OutputTensorAlias(0, input);

    OperatorBase::SetOutputTensor(0, input.Alias());
    return true;
  }

 protected:
  float output_l_;
  float output_h_;
};

} // namespace caffe2

#endif // META_OUTPUT_OP_H_
