#!/usr/bin/env python3

# pyre-fixme[21]: Could not find name `fbnet_building_blocks` in
#  `mobile_cv.arch.fbnet_v2`.
# pyre-fixme[21]: Could not find name `fbnet_blocks_factory` in
#  `mobile_cv.arch.fbnet_v2`.
from . import (
    fbnet_blocks_factory as bf,
    fbnet_building_blocks as bb,
    fbnet_building_blocks_experimental as bbe,
    oct_conv,
)


RESIZE_METHOD_TO_OP = {
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "amp": bb.AdaptiveMaxPool2dConvBNRelu,
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "fmp": bb.FracMaxPool2dConvBNRelu,
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "rnc": bb.ResizeConvBNRelu,
    "rbc": lambda *args, **kwargs:
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    bb.ResizeConvBNRelu(*args, mode="bilinear", **kwargs),
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "rnmp": bb.ResizeMaxPool2dConvBNRelu,
    "rbmp": lambda *args, **kwargs:
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    bb.ResizeMaxPool2dConvBNRelu(*args, mode="bilinear", **kwargs),
    "crn": bbe.ConvBNReluResize,
    "crb": lambda *args, **kwargs: bbe.ConvBNReluResize(
        *args, mode="bilinear", **kwargs
    ),
    "crnmp": bbe.ConvBNReluResizeMaxPool2d,
    "crbmp": lambda *args, **kwargs: bbe.ConvBNReluResizeMaxPool2d(
        *args, mode="bilinear", **kwargs
    ),
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "p3d": bb.Pool3dConvBNRelu,
}


PRIMITIVES_experimental = {
    "oct_ir_k3_0": lambda C_in, C_out, expansion, stride, **kwargs: oct_conv.OctIRFBlock(
        C_in, C_out, expansion, stride, oct_ratio=0.0, **kwargs
    ),
    "oct_irf_k3_0": lambda C_in, C_out, expansion, stride, **kwargs: oct_conv.OctIRFusedConv2D(
        C_in,
        C_out,
        expansion,
        stride,
        oct_ratio=0,
        enable_irf=((False, True), (False, False)),
        **kwargs
    ),
    "oct_irpw_k3_0": lambda C_in, C_out, expansion, stride, **kwargs: oct_conv.OctIRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        oct_ratio=0.0,
        pw_conv=oct_conv.OctConvBNReluSingleUse,
        dw_conv=oct_conv.ConvBNReluAsOct,
        **kwargs
    ),
    "oct_irpw_k3_0.375": lambda C_in, C_out, expansion, stride, **kwargs: oct_conv.OctIRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        oct_ratio=0.375,
        pw_conv=oct_conv.OctConvBNReluSingleUse,
        dw_conv=oct_conv.ConvBNReluAsOct,
        **kwargs
    ),
    "oct_ir_k5_sehsig_0": lambda C_in, C_out, expansion, stride, **kwargs: oct_conv.OctIRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        kernel=5,
        oct_ratio=0.0,
        se={"fc": True, "hsigmoid": True},
        **kwargs
    ),
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "ir_k3_sk": lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in, C_out, expansion, stride, dw_builder=dw_builder_selective_kernel, **kwargs
    ),
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "ir_k3_sk_hs": lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        relu_type="hswish",
        dw_builder=dw_builder_selective_kernel,
        **kwargs
    ),
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "ir_k3_se_sk_hs": lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        relu_type="hswish",
        se={"fc": True, "hsigmoid": True},
        dw_builder=dw_builder_selective_kernel,
        **kwargs
    ),
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "ir_k3_cs2d": lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        dw_conv=lambda *args, **kwargs: bbe.ConvBNReluResize(
            *args, mode="space2depth", **kwargs
        ),
        **kwargs
    ),
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_building_blocks`.
    "ir_k3_s2dc": lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        dw_conv=lambda *args, **kwargs:
        # pyre-fixme[16]: Module `fbnet_v2` has no attribute
        #  `fbnet_building_blocks`.
        bb.ResizeConvBNRelu(*args, mode="space2depth", **kwargs),
        **kwargs
    ),
}


def dw_builder_selective_kernel(channels, stride):
    return bbe.SelectiveKernelConvBNRelu(
        channels, channels, stride=stride, bias=False, group=channels
    )


# pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_blocks_factory`.
bf.add_primitives(PRIMITIVES_experimental)


# modified operations using fractional-stride resize operations


def ir_k5_sehsig_wrapper(op):
    return lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        kernel=5,
        se={"fc": True, "hsigmoid": True},
        dw_conv=op,
        **kwargs
    )


def ir_k5_se_hs_wrapper(op):
    return lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in,
        C_out,
        expansion,
        stride,
        kernel=5,
        relu_type="hswish",
        se={"fc": True, "hsigmoid": True},
        dw_conv=op,
        **kwargs
    )


def ir_k3_wrapper(op):
    return lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in, C_out, expansion, stride, dw_conv=op, **kwargs
    )


def ir_k3_hs_wrapper(op):
    return lambda C_in, C_out, expansion, stride, **kwargs: bb.IRFBlock(
        C_in, C_out, expansion, stride, dw_conv=op, relu_type="hswish", **kwargs
    )


for method, op in RESIZE_METHOD_TO_OP.items():
    ir_k5_sehsig_method = "ir_k5_sehsig_{}".format(method)
    ir_k5_se_hs_method = "ir_k5_se_hs_{}".format(method)
    ir_k3_method = "ir_k3_{}".format(method)
    ir_k3_hs_method = "ir_k3_hs_{}".format(method)
    # pyre-fixme[16]: Module `fbnet_v2` has no attribute `fbnet_blocks_factory`.
    bf.add_primitives(
        {
            ir_k5_sehsig_method: ir_k5_sehsig_wrapper(op),
            ir_k5_se_hs_method: ir_k5_se_hs_wrapper(op),
            ir_k3_method: ir_k3_wrapper(op),
            ir_k3_hs_method: ir_k3_hs_wrapper(op),
        }
    )
