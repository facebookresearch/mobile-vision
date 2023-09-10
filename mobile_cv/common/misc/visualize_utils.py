import math
import os
from typing import Any, Dict, List, Optional

from mobile_cv.common.misc.oss_utils import fb_overwritable

from PIL import Image, ImageDraw, ImageFont


@fb_overwritable()
def get_font_path() -> Optional[str]:
    return None


def save_as_image_grids(
    rows: List[Dict[str, Any]],
    output_dir: str,
    path_manager,
    max_rows_per_image: Optional[int] = None,
    grid_padding_rows: int = 15,
    grid_padding_cols: int = 10,
    font_size: int = 10,
    max_image_size: Optional[int] = None,
) -> List[str]:
    """
    Draw and save image grids, preserve the image sizes.
    rows: List of dicts that represent each row in the grid. Each dict must have
        the keys "images", and optional keys "row_title", "titles", and "labels":
        * images (List[str]): List of image paths for the row,
        * row_title (Optional[str]): The title of the row
        * titles (Optional[List[str]]): List of titles for each image in the row,
            drawing on top of each image
        * labels (Optional[List[str]]): List of labels for each image in the row,
            drawing under each image
    output_dir: output folder of the saved images, image will be saved with file
        names "grid_{start_row_idx}.png"
    path_manager: path manager for io
    max_rows_per_image: maximum number of rows per image, multiple images could
        be generated.
    grid_padding_rows: Number of pixels between each row in the grid
    grid_padding_cols: Number of pixels between each column in the grid
    font_size: Size of font
    max_image_size: maximum image size to fit into the grid
    """
    if not path_manager.exists(output_dir):
        path_manager.mkdirs(output_dir)

    if max_rows_per_image is None:
        max_rows_per_image = len(rows)

    ret = []
    start_row = 0
    while start_row < len(rows):
        end_row = min(len(rows), (start_row + max_rows_per_image))
        grid_img = draw_image_grid_by_rows(
            path_manager,
            rows[start_row:end_row],
            grid_padding_rows=grid_padding_rows,
            grid_padding_cols=grid_padding_cols,
            font_size=font_size,
        )

        output_filepath = os.path.join(output_dir, f"grid_{start_row:05d}.png")
        with path_manager.open(output_filepath, "wb") as fp:
            grid_img.save(fp)
        start_row = end_row
        ret.append(output_filepath)

    return ret


def draw_image_grid_by_rows(
    path_manager,
    rows: List[Dict[str, Any]],
    grid_padding_rows: int = 15,
    grid_padding_cols: int = 10,
    font_size: int = 10,
    max_image_size: Optional[int] = None,
) -> Image.Image:
    """
    Draw a grid of images into a single image, preserve the image sizes.
    rows: List of dicts that represent each row in the grid. Each dict must have
        the keys "images", and optional keys "row_title", "titles", and "labels":
        * images (List[str]): List of image paths for the row,
        * row_title (Optional[str]): The title of the row
        * titles (Optional[List[str]]): List of titles for each image in the row,
            drawing on top of each image
        * labels (Optional[List[str]]): List of labels for each image in the row,
            drawing under each image
    grid_padding_rows: Number of pixels between each row in the grid
    grid_padding_cols: Number of pixels between each column in the grid
    font_size: Size of font
    """
    image_paths = []
    columns: Optional[int] = None
    row_titles = []
    image_titles = []
    image_labels = []
    for item in rows:
        # images
        images = item["images"]
        assert isinstance(images, list)
        if columns is None:
            columns = len(images)
        else:
            assert len(images) == columns
        image_paths.extend(images)

        # row title
        row_title = item.get("row_title", None)
        row_titles.append(row_title)

        # image_titles
        image_title = item.get("titles", None)
        if image_title is not None:
            assert isinstance(image_title, list) and len(image_title) == columns
            image_titles.extend(image_title)
        else:
            image_titles.extend([None] * columns)

        # image_labels
        image_label = item.get("labels", None)
        if image_label is not None:
            assert isinstance(image_label, list) and len(image_label) == columns
            image_labels.extend(image_label)
        else:
            image_labels.extend([None] * columns)

    return draw_image_grid(
        path_manager,
        image_paths,
        columns=columns,
        row_titles=row_titles,
        image_titles=image_titles,
        image_labels=image_labels,
        grid_padding_rows=grid_padding_rows,
        grid_padding_cols=grid_padding_cols,
        font_size=font_size,
        max_image_size=max_image_size,
    )


def draw_image_grid(
    path_manager,
    image_paths: List[str],
    columns: int,
    row_titles: Optional[List[str]] = None,
    image_titles: Optional[List[str]] = None,
    image_labels: Optional[List[str]] = None,
    grid_padding_rows: int = 15,
    grid_padding_cols: int = 10,
    font_size: int = 10,
    max_image_size: Optional[int] = None,
) -> Image.Image:
    """
    Draw a grid of images into a single image, preserve the image sizes.
    image_paths: List of images paths
    columns: number of columns in the grid
    """

    num_images = len(image_paths)
    rows = math.ceil(num_images / columns)

    if row_titles is not None:
        assert len(row_titles) == rows, f"{len(row_titles), rows}"
    if image_titles is not None:
        assert len(image_titles) == num_images, f"{len(image_titles), num_images}"
    if image_labels is not None:
        assert len(image_labels) == num_images, f"{len(image_labels), num_images}"

    # Load the images from file paths
    images = []
    for ip in image_paths:
        if ip is None:
            empty_size = max_image_size or 25
            cur = Image.new(mode="RGB", size=(empty_size, empty_size), color="white")
        else:
            with path_manager.open(ip, "rb") as fp:
                cur = Image.open(fp)
                cur.load()
            if max_image_size is not None:
                cur.thumbnail((max_image_size, max_image_size), Image.LANCZOS)
        images.append(cur)

    # Get the dimensions of the images to create the grid
    width_per_column, height_per_row = _get_grid_dimensions(images, columns)

    # Calculate grid dimensions
    # only pad between columns
    grid_width = sum(width_per_column) + grid_padding_cols * (columns - 1)
    # each row always have grid_padding_rows * 3 padding (2 for title and image
    # title at the top, one for image label at the bottom)
    grid_height = sum(height_per_row) + (grid_padding_rows * 3) * rows

    # Create a blank canvas for the grid
    grid_image = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid_image)

    # # Load fonts
    font_path = get_font_path()
    if font_path is not None:
        font_path = path_manager.get_local_path(font_path)
        title_font = ImageFont.truetype(font_path, font_size)
        label_font = ImageFont.truetype(font_path, font_size)
    else:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()

    # Draw the images and labels on the grid
    for row in range(rows):
        # Calculate starting position for the current row
        start_x = 0
        start_y = sum(height_per_row[:row]) + row * (grid_padding_rows * 3)

        # Draw the title for the current row
        if row_titles is not None:
            draw.text(
                (start_x, start_y), row_titles[row], font=title_font, fill="black"
            )

        # Draw the images and labels for the current row
        for col in range(columns):
            # Calculate the position for the current image and label
            image_index = row * columns + col
            if image_index >= num_images:
                break

            image = images[image_index]
            label = image_labels[image_index] if image_labels is not None else None
            image_title = (
                image_titles[image_index] if image_titles is not None else None
            )

            image_x = start_x + sum(width_per_column[:col]) + grid_padding_cols * col
            image_y = start_y + grid_padding_rows * 2

            # Paste the image onto the grid
            if image is not None:
                grid_image.paste(image, (image_x, image_y))

            # Draw the label below the image
            if label is not None:
                label_width, label_height = label_font.getsize(label)
                label_x = (
                    image_x + (width_per_column[col] - label_width) // 2
                )  # Center horizontally
                label_y = image_y + height_per_row[row]

                draw.text((label_x, label_y), label, font=label_font, fill="black")

            # Draw the image title above the image
            if image_title is not None:
                image_title_width, image_title_height = title_font.getsize(image_title)
                image_title_x = (
                    image_x + (width_per_column[col] - image_title_width) // 2
                )  # Center horizontally
                image_title_y = start_y + grid_padding_rows

                draw.text(
                    (image_title_x, image_title_y),
                    image_title,
                    font=title_font,
                    fill="black",
                )

    return grid_image


def _get_grid_dimensions(images: List[Image.Image], num_columns: int):
    assert (
        len(images) % num_columns == 0
    ), f"Number of images {len(images)} is not divisible by {num_columns}"
    num_rows = len(images) // num_columns

    ret_width_per_column = [0.0] * num_columns
    ret_height_per_row = [0.0] * num_rows

    for idx, image in enumerate(images):
        ri, rj = divmod(idx, num_columns)
        width, height = image.size
        if width > ret_width_per_column[rj]:
            ret_width_per_column[rj] = width
        if height > ret_height_per_row[ri]:
            ret_height_per_row[ri] = height

    return ret_width_per_column, ret_height_per_row
