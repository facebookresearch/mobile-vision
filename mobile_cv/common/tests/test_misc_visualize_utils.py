# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import tempfile
import unittest

import mobile_cv.common.misc.visualize_utils as vu
import numpy as np
import PIL
from mobile_cv.common.utils_io import get_path_manager


SAVE_IMAGE = True


def _create_image(output_dir, idx, height, width, color=None):
    os.makedirs(output_dir, exist_ok=True)
    ret = os.path.join(output_dir, f"test_image_{idx}.png")
    if height is None or width is None:
        return None
    PIL.Image.new("RGB", (width, height), color=color).save(ret)
    return ret


class TestMiscVisualizationUtils(unittest.TestCase):
    def test_draw_image_grid(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/common:tests -- -r test_draw_image_grid$
        """
        pm = get_path_manager()

        num_images = 12
        num_columns = 4
        image_size = 64

        with tempfile.TemporaryDirectory() as output_dir:
            images = [
                _create_image(
                    output_dir,
                    idx,
                    image_size,
                    image_size,
                    (idx * 20, idx * 20, idx * 20),
                )
                for idx in range(num_images)
            ]
            ret = vu.draw_image_grid(
                pm,
                images,
                columns=num_columns,
                grid_padding_rows=0,
                grid_padding_cols=0,
            )
            self.assertEqual(ret.height, image_size * (num_images // num_columns))
            self.assertEqual(ret.width, image_size * num_columns)

            img = np.array(ret)
            for row in range(3):
                for col in range(num_columns):
                    rs = row * image_size
                    cs = col * image_size
                    self.assertEqual(
                        np.linalg.norm(
                            img[rs : rs + image_size, cs : cs + image_size, :]
                            - ((row * num_columns + col) * 20)
                        ),
                        0.0,
                    )

            if SAVE_IMAGE:
                with pm.open(os.path.join(output_dir, "test_vis.png"), "wb") as fp:
                    ret.save(fp)
                print(output_dir)

    def test_draw_image_grid_text(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/common:tests -- -r test_draw_image_grid_text$
        """

        pm = get_path_manager()

        num_images = 6
        num_columns = 3
        num_rows = num_images // num_columns
        image_size = 64

        with tempfile.TemporaryDirectory() as output_dir:
            images = [
                _create_image(
                    output_dir,
                    idx,
                    image_size,
                    image_size,
                    (idx * 30, idx * 30, idx * 30),
                )
                for idx in range(num_images)
            ]
            row_titles = [f"title_{x}" for x in range(num_rows)]
            image_titles = [f"image_{x}" for x in range(num_images)]
            image_labels = [f"label_{x}" for x in range(num_images)]
            ret = vu.draw_image_grid(
                pm,
                images,
                columns=num_columns,
                row_titles=row_titles,
                image_titles=image_titles,
                image_labels=image_labels,
                grid_padding_rows=12,
                grid_padding_cols=5,
                font_size=10,
            )
            self.assertEqual(ret.height, (image_size + 12 * 3) * num_rows)
            self.assertEqual(
                ret.width, image_size * num_columns + 5 * (num_columns - 1)
            )

            img = np.array(ret)
            for row in range(num_rows):
                for col in range(num_columns):
                    rs = row * (image_size + 12 * 3) + 12 * 2
                    cs = col * (image_size + 5)
                    self.assertEqual(
                        np.linalg.norm(
                            img[rs : rs + image_size, cs : cs + image_size, :]
                            - ((row * num_columns + col) * 30)
                        ),
                        0.0,
                    )

            if SAVE_IMAGE:
                with pm.open(os.path.join(output_dir, "test_vis.png"), "wb") as fp:
                    ret.save(fp)
                print(output_dir)

    def test_save_as_image_grids(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/common:tests -- -r test_save_as_image_grids$
        """
        pm = get_path_manager()

        num_columns = 3
        num_rows = 10
        image_size = 32

        with tempfile.TemporaryDirectory() as output_dir:
            src_output_dir = os.path.join(output_dir, "src")
            rows = [
                {
                    "images": [
                        _create_image(src_output_dir, idx, image_size, image_size)
                        for idx in range(num_columns)
                    ],
                    "row_title": "row_title",
                    "titles": [f"title_{idx}" for idx in range(num_columns)],
                    "labels": [f"label_{idx}" for idx in range(num_columns)],
                }
            ] * num_rows

            dst_output_dir = os.path.join(output_dir, "dst")
            ret = vu.save_as_image_grids(
                rows=rows,
                output_dir=dst_output_dir,
                path_manager=pm,
                max_rows_per_image=4,
            )
            self.assertEqual(len(ret), 3)
            for img_path in ret:
                self.assertTrue(os.path.exists(img_path))

    def test_draw_image_grid_text_resolution(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/common:tests -- -r test_draw_image_grid_text_resolution$
        """

        pm = get_path_manager()

        num_images = 6
        num_columns = 3
        num_rows = num_images // num_columns
        image_heights = [64, 70, 80, 85, 50, 60]
        image_widths = [64, 70, 40, 50, 50, 70]
        row_heights = [80, 85]
        col_widths = [64, 70, 70]

        with tempfile.TemporaryDirectory() as output_dir:
            images = [
                _create_image(
                    output_dir,
                    idx,
                    image_heights[idx],
                    image_widths[idx],
                    (idx * 30, idx * 30, idx * 30),
                )
                for idx in range(num_images)
            ]
            row_titles = [f"title_{x}" for x in range(num_rows)]
            image_titles = [f"image_{x}" for x in range(num_images)]
            image_labels = [f"label_{x}" for x in range(num_images)]
            ret = vu.draw_image_grid(
                pm,
                images,
                columns=num_columns,
                row_titles=row_titles,
                image_titles=image_titles,
                image_labels=image_labels,
                grid_padding_rows=12,
                grid_padding_cols=5,
                font_size=10,
            )
            self.assertEqual(ret.height, 80 + 85 + 12 * 3 * 2)
            self.assertEqual(ret.width, 64 + 70 + 70 + 5 * 2)

            img = np.array(ret)
            for row in range(num_rows):
                rs = sum(row_heights[:row]) + 12 * 3 * row + 12 * 2
                for col in range(num_columns):
                    cs = sum(col_widths[:col]) + 5 * col
                    idx = row * num_columns + col
                    self.assertEqual(
                        np.linalg.norm(
                            img[
                                rs : rs + image_heights[idx],
                                cs : cs + image_widths[idx],
                                :,
                            ]
                            - (idx * 30)
                        ),
                        0.0,
                    )

            if SAVE_IMAGE:
                with pm.open(os.path.join(output_dir, "test_vis.png"), "wb") as fp:
                    ret.save(fp)
                print(output_dir)

    def test_draw_image_grid_text_resolution_empty(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/common:tests -- -r test_draw_image_grid_text_resolution_empty$
        """

        pm = get_path_manager()

        num_images = 10
        num_columns = 5
        num_rows = num_images // num_columns
        image_heights = [64, None, 80, None, None, 85, 50, None, 64, None]
        image_widths = [64, None, 40, None, None, 50, 50, None, 64, None]

        with tempfile.TemporaryDirectory() as output_dir:
            images = [
                _create_image(
                    output_dir,
                    idx,
                    image_heights[idx],
                    image_widths[idx],
                    (idx * 20, idx * 20, idx * 20),
                )
                for idx in range(num_images)
            ]
            row_titles = [f"title_{x}" for x in range(num_rows)]
            image_titles = [f"image_{x}" for x in range(num_images)]
            image_labels = [
                f"label_{x}" if image_heights[x] is not None else None
                for x in range(num_images)
            ]
            ret = vu.draw_image_grid(
                pm,
                images,
                columns=num_columns,
                row_titles=row_titles,
                image_titles=image_titles,
                image_labels=image_labels,
                grid_padding_rows=12,
                grid_padding_cols=5,
                font_size=10,
            )
            self.assertEqual(ret.height, 80 + 85 + 12 * 3 * 2)
            # 25 is the default image size when the whole column is empty
            self.assertEqual(ret.width, 64 + 50 + 40 + 64 + 25 + 5 * 4)

            if SAVE_IMAGE:
                with pm.open(os.path.join(output_dir, "test_vis.png"), "wb") as fp:
                    ret.save(fp)
                print(output_dir)

    def test_draw_image_grid_text_resolution_empty_max_size(self):
        """
        buck2 run @mode/dev-nosan //mobile-vision/mobile_cv/mobile_cv/common:tests -- -r test_draw_image_grid_text_resolution_empty_max_size$
        """

        pm = get_path_manager()

        num_images = 6
        num_columns = 3
        num_rows = num_images // num_columns
        image_heights = [64, None, 80, 85, 50, None]
        image_widths = [64, None, 40, 50, 50, None]

        with tempfile.TemporaryDirectory() as output_dir:
            images = [
                _create_image(
                    output_dir,
                    idx,
                    image_heights[idx],
                    image_widths[idx],
                    (idx * 20, idx * 20, idx * 20),
                )
                for idx in range(num_images)
            ]
            row_titles = [f"title_{x}" for x in range(num_rows)]
            image_titles = [f"image_{x}" for x in range(num_images)]
            image_labels = [
                f"label_{x}" if image_heights[x] is not None else None
                for x in range(num_images)
            ]
            ret = vu.draw_image_grid(
                pm,
                images,
                columns=num_columns,
                row_titles=row_titles,
                image_titles=image_titles,
                image_labels=image_labels,
                grid_padding_rows=12,
                grid_padding_cols=5,
                font_size=10,
                max_image_size=45,
            )
            self.assertEqual(ret.height, 45 * 2 + 12 * 3 * 2)
            self.assertEqual(ret.width, 45 * 3 + 5 * 2)

            if SAVE_IMAGE:
                with pm.open(os.path.join(output_dir, "test_vis.png"), "wb") as fp:
                    ret.save(fp)
                print(output_dir)
