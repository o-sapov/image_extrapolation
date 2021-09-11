# -*- coding: utf-8 -*-
"""prepare_data.py

Author -- Oleksii Sapov
Contact -- sapov@gmx.at
Date -- 25.07.2021


###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This material, no matter whether in printed or electronic form, may be used for
personal and non-commercial educational use only. Any reproduction of this
manuscript, no matter whether as a whole or in parts, no matter whether in
printed or in electronic form, requires explicit prior acceptance of the
authors.

###############################################################################

Pipeline for processing images.
"""

import hashlib
import os
import glob
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import dill as pkl
import random

IN_FOLDER = os.path.join('.', 'in')
OUT_FOLDER = os.path.join('.', 'out')


def get_filepaths(inp_dir_absolute):
    """ Creates a sorted list of files from inp_dir recursively"""

    filepaths = sorted(glob.glob(os.path.join(inp_dir_absolute, '**'), recursive=True))
    filepaths = [f for f in filepaths if os.path.isfile(f)]
    return filepaths


def rgb2gray(rgb_array: np.ndarray, r=0.2989, g=0.5870, b=0.1140):
    """Convert numpy array with 3 color channels of shape (..., 3) to grayscale
    Disclaimer:
    Author -- Michael Widrich
    Contact -- widrich@ml.jku.at
    Date -- 01.02.2020
    """
    grayscale_array = (rgb_array[..., 0] * r +
                       rgb_array[..., 1] * g +
                       rgb_array[..., 2] * b)
    grayscale_array = np.round(grayscale_array)
    grayscale_array = np.asarray(grayscale_array, dtype=np.uint8)
    return grayscale_array


def write_log(logfile_path, logfile_content):
    with open(logfile_path, 'w', newline='\n') as fh:
        for b in logfile_content:
            print(b, file=fh)


def validation():
    print("Creating list of valid JPEG files.")
    inp_dir_absolute = os.path.abspath(IN_FOLDER)
    files = get_filepaths(inp_dir_absolute)
    # list of strings, e.g. ['folder0;1', 'folder1;5']
    logfile_content = []
    # list of valid files
    files_to_copy = []
    # hashes to compare if the file was already processed
    hashes = []

    # Validation check: add boolean value for every rule. Than check if there is any False
    for file in files:
        validation = []
        # 1 The file name ends with .jpg, .JPG, .JPEG, or .jpeg.
        if file.endswith(('.jpg', '.JPG', '.JPEG', '.jpeg')):
            validation.append(True)
        else:
            validation.append(False)
        # 2 The file size is larger than 10kB.
        if os.path.getsize(file) > 1e4:
            validation.append(True)
        else:
            validation.append(False)
        # 3  The file can be read as image (i.e. the PIL/pillow module does not raise an exception
        # when reading the file).
        try:
            image = Image.open(file)  # This returns a PIL image
            image = np.array(image)  # We can convert it to a numpy array
            validation.append(True)
        except:
            # to avoid error later
            image = np.array([1])
            validation.append(False)

        # 5 The image data does have variance> 0, i.e. there is not just 1 value in the image data.
        if len(np.unique(image)) > 1:
            validation.append(True)
        else:
            validation.append(False)

        # 6 The same image data has not been copied already.
        hashing_function = hashlib.sha256()
        hashing_function.update(image.tobytes())
        file_hash = hashing_function.digest()

        if file_hash not in hashes:
            validation.append(True)
        else:
            validation.append(False)

        if not False in validation:
            files_to_copy.append(file)
            hashes.append(file_hash)

        for index, item in enumerate(validation):
            if item == False:
                file_path = os.path.relpath(file, inp_dir_absolute)
                record = file_path + ';' + str(index + 1)
                logfile_content.append(record)
                break

    return files_to_copy, logfile_content


def valid_jpeg_files():
    valid_images, logfile_content = validation()
    print("Writing log with invalid files.")
    write_log(os.path.join(OUT_FOLDER, 'invalid_files.log'), logfile_content)

    return valid_images


def convert2greyscaled(pil_images):
    print("Converting to greyscale.")
    greyscaled_images = []
    for image in pil_images:
        image = np.array(image)
        if len(image.shape) == 2 and image.shape[0] >= 100 and image.shape[1] >= 100:
            greyscaled_images.append(image)
        else:
            image = rgb2gray(image)
            greyscaled_images.append(image)

    return greyscaled_images


def reduce_image_sizes(valid_jpeg_files):
    """This script will take a folder input_path and rescale all .jpg images in it to
a resolution such that the file size is ~850kB or smaller.
Note that the new file size is typically much smaller due to .jpg compression
but that will not bother us as long as the content is not too blurry.

Disclaimer:
Author -- Michael Widrich
Contact -- widrich@ml.jku.at
Date -- 01.03.2020
"""
    print("Reducing image size.")
    desired_file_size = 850e3
    image_files = valid_jpeg_files
    images_list = []

    for image_file in image_files:
        # Get size of file in Bytes
        file_size = os.path.getsize(image_file)
        # Calculate reduction factor
        reduction_factor = desired_file_size / file_size
        # We want to apply rescaling factor to x and y dimension -> 2D -> we need to apply the square-root
        reduction_factor = reduction_factor ** (1. / 2)
        # Read file
        image_file = Image.open(image_file)
        # Only change image resolution if we need to reduce the file size
        if reduction_factor < 1:
            # Get current resolution
            old_size = image_file.size
            # Calculate new file-size
            new_size = [int(s * reduction_factor) for s in old_size]
            # Resize to new_size and save image in new folder
            new_image = image_file.resize(new_size)
            images_list.append(new_image)
        else:
            # If the file size is already small enough, we can just copy the file
            images_list.append(image_file)
    return images_list


def resize_90by90(images, im_shape=90):
    print("Reducing images shape to 90 by 90.")
    resized_images = []

    resize_transforms = transforms.Compose([
        transforms.Resize(size=im_shape),
        transforms.CenterCrop(size=(im_shape, im_shape)),
    ])

    for image in images:
        image = Image.fromarray(image)
        image = resize_transforms(image)
        image = np.array(image)
        resized_images.append(image)
    return resized_images


def ex4(image_array: np.ndarray, border_x: tuple, border_y: tuple):
    """Creates two input arrays and one target array from one input image
    • The area of known pixels in the test set images will be at least (75, 75) pixels large.
• The borders containing unknown pixels in the test set images will be at least 5 pixels
wide on each side.
    """

    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise NotImplementedError("image_array must be a 2D numpy array")

    border_x_start, border_x_end = border_x
    border_y_start, border_y_end = border_y

    try:  # Check for conversion to int (would raise ValueError anyway but we will write a nice error message)
        border_x_start = int(border_x_start)
        border_x_end = int(border_x_end)
        border_y_start = int(border_y_start)
        border_y_end = int(border_y_end)
    except ValueError as e:
        raise ValueError(f"Could not convert entries in border_x and border_y ({border_x} and {border_y}) to int! "
                         f"Error: {e}")

    if border_x_start < 1 or border_x_end < 1:
        raise ValueError(f"Values of border_x must be greater than 0 but are {border_x_start, border_x_end}")

    if border_y_start < 1 or border_y_end < 1:
        raise ValueError(f"Values of border_y must be greater than 0 but are {border_y_start, border_y_end}")

    remaining_size_x = image_array.shape[0] - (border_x_start + border_x_end)
    remaining_size_y = image_array.shape[1] - (border_y_start + border_y_end)
    if remaining_size_x < 16 or remaining_size_y < 16:
        raise ValueError(f"the size of the remaining image after removing the border must be greater equal (16,16) "
                         f"but was ({remaining_size_x},{remaining_size_y})")

    # Create known_array
    known_array = np.zeros_like(image_array)
    known_array[border_x_start:-border_x_end, border_y_start:-border_y_end] = 1

    # Create target_array - don't forget to use .copy(), otherwise target_array and image_array might point to the
    # same array!
    target_array = image_array[known_array == 0].copy()

    # Use image_array as input_array
    image_array[known_array == 0] = 0

    return image_array, known_array, target_array


def compress(images):
    print("Compressing and writing to pickle file.")
    input_arrays = []
    known_arrays = []
    borders_x = []
    borders_y = []
    targets = []
    sample_ids = []

    for idx, image in enumerate(images):
        # the values for the borders should be between 5 and 9 both included
        random_border_x = [random.randint(5, 9) for i in range(2)]
        borders_x.append(np.array(random_border_x, dtype=np.int64))

        random_border_y = [random.randint(5, 9) for i in range(2)]
        borders_y.append(np.array(random_border_y, dtype=np.int64))

        image = np.copy(image)
        input_array, known_array, target = ex4(image_array=image, border_x=tuple(random_border_x),
                                               border_y=tuple(random_border_y))
        input_arrays.append(input_array)
        known_arrays.append(known_array)
        targets.append(target)
        sample_ids.append(idx)

    img_objects = dict(input_arrays=tuple(input_arrays), known_arrays=tuple(known_arrays),
                       borders_x=tuple(borders_x), borders_y=tuple(borders_y),
                       sample_ids=tuple(sample_ids))

    filename_dataset = os.path.join(OUT_FOLDER, 'dataset.pkl')
    with open(filename_dataset, 'wb') as f:
        pkl.dump(img_objects, f)

    filename_targets = os.path.join(OUT_FOLDER, 'targets.pkl')
    with open(filename_targets, 'wb') as f:
        pkl.dump(targets, f)


if __name__ == "__main__":
    images = valid_jpeg_files()
    images = reduce_image_sizes(images)
    images = convert2greyscaled(images)
    images = resize_90by90(images)
    compress(images)
