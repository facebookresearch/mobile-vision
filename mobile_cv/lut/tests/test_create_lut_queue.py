from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tempfile
import unittest

import mobile_cv.lut.lib.lut_ops as lut_ops
import mobile_cv.lut.lib.lut_schema as lut


def create_ops():
    ret = []
    item1 = lut.LutItem(
        op=lut_ops.Conv2d(
            3, 20, 3, stride=(2, 2), padding=(1, 1), dilation=1, groups=1, bias=False
        ),
        input_shapes=[[1, 3, 224, 224]],
    )
    item2 = lut.LutItem(
        op=lut_ops.Conv2d(
            20, 20, 5, stride=1, padding=0, dilation=(1, 1), groups=2, bias=False
        ),
        input_shapes=[[1, 20, 224, 224]],
    )
    item3 = lut.LutItem(
        op=lut_ops.ConvTranspose2d(
            20,
            20,
            5,
            stride=2,
            padding=1,
            output_padding=1,
            dilation=(1, 1),
            groups=2,
            bias=False,
        ),
        input_shapes=[[1, 20, 224, 224]],
    )
    ret = [item1, item2, item3]
    return ret


def get_name():
    return None


def get_config(lut_file):
    ret = {
        "lut_file": lut_file,
        "device": "S8",
    }
    return ret


class TestCreateLutQueue(unittest.TestCase):
    def test_create_lut_queue(self):
        import mobile_cv.lut.lib.create_lut_queue as create_lut_queue

        # TODO: use make_temp_directory in mobile_cv
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = os.path.join(tmp_dir, "{user}/lut")
            args_list = [
                "--create_script",
                __file__,
                "--output_dir",
                out_dir,
            ]
            out_file, out_config_file = create_lut_queue.main(args_list)

            self.assertTrue(os.path.isfile(out_file))
            self.assertTrue(os.path.isfile(out_config_file))


if __name__ == "__main__":
    unittest.main()
