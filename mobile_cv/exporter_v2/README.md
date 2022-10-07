# Exporter V2

A PyTorch oriented library for model export.

## Concepts

* `export`: The entire process of turning PyTorch model(s) into deployable format(s). It contains two steps: 1): **convert** and 2): **serialization**.
    * `export_target`: A string given by user to describe the export process, in the format of **{backend}_{convert_target}@{additional_flags}**. For examples: "torchscript_quantized@mobile" means converting to "quantized" (**convert_target**) model, saving as "torchscript" (**backend**) and optimizing for "mobile".
    * `additional_flags`: Extra flags for **serialization** step.
* `convert`: Converting the PyTorch model into another PyTorch model (ideally TorchScript) that are ready to be serizalized for deployment. It usually involves changes of the semantics and numerics from the original PyTorch model, such as applying quantization, wrapping model, etc. It also includes deployment related "conversion" like tracing, scripting or delegating to specific backend. The **convert** step is customized by user using **Converter** interface.
    * `convert_target`: An arbitrary string describing the semantic nature of converted model, defined by user. Examples are "quantized", "int8", "hybrid", "benchmark", "predictor", "ios", etc.
* `serialization`: Saving the output of **convert** step to certain **backend** (deployable format). The library support a fixed set of serialization **backends** such as TorchScript, ONNX, etc.
    * `backend`: A string describing the type of backend, examples are "torchscript", "onnx", etc.
