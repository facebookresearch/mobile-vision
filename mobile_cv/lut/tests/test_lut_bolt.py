from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tempfile
import unittest

import mobile_cv.lut.lib.fb.bolt.lut_bolt as lut_bolt
import mobile_cv.lut.lib.fb.bolt.utils_bolt as utils_bolt
import mobile_cv.lut.lib.lut_ops as lut_ops
import mobile_cv.lut.lib.lut_schema as lut_schema
import mobile_cv.lut.lib.utils as utils
import numpy as np


# fmt: off
TEST_LUT = [
    {
        "type": "conv-1-96-96-544-1-3-1-544-1-1-1-0",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 12771400.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "dwise_conv-1-96-96-928-928-5-5-1-4-4-2-2",
        "op": "OP_Caffe2_QuantizedDepthwiseConv_8x8to8",
        "value": 5697160.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "conv-1-96-96-864-1-3-3-864-4-4-1-1",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 4047430.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "conv-1-3-3-768-960-3-3-768-1-1-1-1",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 2743590.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "dwise_conv-1-50-50-416-416-5-5-1-2-2-2-2",
        "op": "OP_Caffe2_QuantizedDepthwiseConv_8x8to8",
        "value": 2642200.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "conv-1-12-12-768-928-3-3-768-4-4-1-1",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 2611180.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "conv-1-6-6-608-800-5-5-608-4-4-2-2",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 2517690.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "conv-1-3-3-928-1024-5-5-928-4-4-2-2",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 2478750.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "conv-1-6-6-480-544-5-5-480-4-4-2-2",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 1353190.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
    {
        "type": "conv-1-3-3-448-864-3-3-448-1-1-1-1",
        "op": "OP_Caffe2_QuantizedConv2d_8x8to8",
        "value": 1164330.0,
        "unit": "cycle",
        "metric": "node-cycles"
    },
]
# fmt: on


class TestLutBolt(unittest.TestCase):
    """Lut table in bolt nn format"""

    def test_lut_bolt(self):
        item1 = lut_schema.OpInfo(
            op=lut_ops.Conv2d(
                608,
                800,
                5,
                stride=(4, 4),
                padding=(2, 2),
                dilation=1,
                groups=1,
                bias=False,
            ),
            input_shapes=[[2, 608, 6, 6]],
        )
        item2 = lut_schema.OpInfo(
            op=lut_ops.Conv2d(
                416,
                416,
                5,
                stride=2,
                padding=2,
                dilation=(1, 1),
                groups=416,
                bias=False,
            ),
            input_shapes=[[3, 416, 50, 50]],
        )
        ops = [item1, item2]

        fd, path = tempfile.mkstemp()
        try:
            utils.save_to_json_file(TEST_LUT, path)
            lut = lut_bolt.LutBolt(path)

            latencies = lut.query(ops)
            np.testing.assert_almost_equal(
                np.array(latencies), np.array([0.00251769 * 2.0, 0.0026422 * 3.0])
            )
        finally:
            os.remove(path)

    def test_lut_bolt_approx(self):
        HW_SCALE = 2
        BATCH = 3
        item2 = lut_schema.OpInfo(
            op=lut_ops.Conv2d(
                416,
                416,
                5,
                stride=2,
                padding=2,
                dilation=(1, 1),
                groups=416,
                bias=False,
            ),
            input_shapes=[[BATCH, 416, 50 * HW_SCALE, 50 * HW_SCALE]],
        )
        ops = [item2]

        fd, path = tempfile.mkstemp()
        try:
            utils.save_to_json_file(TEST_LUT, path)
            lut = lut_bolt.LutBolt(path)

            latencies = lut.query(ops)
            np.testing.assert_almost_equal(
                np.array(latencies), np.array([0.0026422 * HW_SCALE * HW_SCALE * BATCH])
            )
        finally:
            os.remove(path)

    def test_lut_convert_to_bolt(self):
        item1 = lut_schema.OpInfo(
            op=lut_ops.Conv2d(
                608,
                800,
                5,
                stride=(4, 4),
                padding=(2, 2),
                dilation=1,
                groups=1,
                bias=False,
            ),
            input_shapes=[[1, 608, 6, 6]],
        )
        item2 = lut_schema.OpInfo(
            op=lut_ops.Conv2d(
                416,
                416,
                5,
                stride=2,
                padding=2,
                dilation=(1, 1),
                groups=416,
                bias=False,
            ),
            input_shapes=[[1, 416, 50, 50]],
        )
        ops = [item1, item2]

        GT_STRS = [
            "conv 1 6 6 608 800 5 5 608 4 4 2 2",
            "dwise_conv 1 50 50 416 416 5 5 1 2 2 2 2",
        ]

        bolt_strs = utils_bolt.convert_ops_to_bolt(ops)
        self.assertEqual(bolt_strs, GT_STRS)

    def test_convert_bolt(self):
        item1 = lut_schema.OpInfo(
            op=lut_ops.Conv2d(
                608,
                800,
                5,
                stride=(4, 4),
                padding=(2, 2),
                dilation=1,
                groups=1,
                bias=False,
            ),
            input_shapes=[[2, 608, 6, 6]],
        )
        item2 = lut_schema.OpInfo(
            op=lut_ops.Conv2d(
                416,
                416,
                5,
                stride=2,
                padding=2,
                dilation=(1, 1),
                groups=416,
                bias=False,
            ),
            input_shapes=[[3, 416, 50, 50]],
        )
        item3 = lut_schema.OpInfo(
            op=lut_ops.ConvTranspose2d(
                416,
                416,
                5,
                stride=2,
                padding=2,
                output_padding=2,
                groups=1,
                bias=False,
            ),
            input_shapes=[[3, 416, 50, 50]],
        )
        ops = [item1, item2, item3]
        bolt_items = utils_bolt.convert_ops_to_bolt(ops)
        # op str after conversion using '-' as delimiter
        bolt_items = [x.replace(" ", "-") for x in bolt_items]

        recon_ops = []
        for x in bolt_items:
            recon_ops.append(lut_schema.OpInfo(*utils_bolt.convert_to_op(x)))

        self.assertEqual(ops, recon_ops)


if __name__ == "__main__":
    unittest.main()
