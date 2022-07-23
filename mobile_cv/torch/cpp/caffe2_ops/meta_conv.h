#ifndef META_CONV_OP_H_
#define META_CONV_OP_H_

#include "caffe2/caffe2/core/context.h"
#include "caffe2/caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(MetaConv)

namespace caffe2 {

template <class Context>
class MetaConvOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit MetaConvOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        weight_l_(this->template GetSingleArgument<float>("weight_l", 1.)),
        weight_h_(this->template GetSingleArgument<float>("weight_h", 1.)),
        bias_l_(this->template GetSingleArgument<float>("bias_l", 1.)),
        bias_h_(this->template GetSingleArgument<float>("bias_h", 1.)),
        output_l_(this->template GetSingleArgument<float>("output_l", 1.)),
        output_h_(this->template GetSingleArgument<float>("output_h", 1.)) {
    CAFFE_ENFORCE(OperatorBase::HasArgument("weight_l"), "Requires weight_l");
    CAFFE_ENFORCE(OperatorBase::HasArgument("weight_h"), "Requires weight_h");
    CAFFE_ENFORCE(OperatorBase::HasArgument("bias_l"), "Requires bias_l");
    CAFFE_ENFORCE(OperatorBase::HasArgument("bias_h"), "Requires bias_h");
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
  float weight_l_;
  float weight_h_;
  float bias_l_;
  float bias_h_;
  float output_l_;
  float output_h_;
};

} // namespace caffe2

#endif // META_CONV_OP_H_
