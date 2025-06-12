# -*- coding: utf-8 -*-
"""
STEFANN Inference Script | Scene Text Editor using Font Adaptive Neural Network
Based on the original STEFANN project by Prasun Roy
GitHub: https://github.com/prasunroy/stefann
"""

import argparse
import os
import numpy as np
import cv2
from PIL import Image
from keras.models import model_from_json

# Ensure keras using tensorflow backend
os.environ["KERAS_BACKEND"] = "tensorflow"


def load_models(models_path="models"):
    """Load neural network models"""
    try:
        with open(os.path.join(models_path, "fannet.json"), "r") as fp:
            net_f = model_from_json(fp.read())
        with open(os.path.join(models_path, "colornet.json"), "r") as fp:
            net_c = model_from_json(fp.read())

        net_f.load_weights(os.path.join(models_path, "fannet_weights.h5"))
        net_c.load_weights(os.path.join(models_path, "colornet_weights.h5"))

        return net_f, net_c
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None


def opencv_version():
    """Get OpenCV version number"""
    return int(cv2.__version__.split(".")[0])


def binarize(image, points=None, thresh=150, maxval=255, thresh_type=0):
    """Binarize image for text detection"""
    # Convert image to grayscale if needed
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image.copy()

    # Region selection if points are provided
    if points is not None and len(points) > 2:
        points = np.array(points, np.int64)
        mask = np.zeros_like(image, np.uint8)
        cv2.fillConvexPoly(mask, points, (255, 255, 255), cv2.LINE_AA)
        image = cv2.bitwise_and(image, mask)

    # Estimate mask 1 from MSER
    msers = cv2.MSER_create().detectRegions(image)[0]
    setyx = set()
    for region in msers:
        for point in region:
            setyx.add((point[1], point[0]))
    setyx = tuple(np.transpose(list(setyx)))
    mask1 = np.zeros(image.shape, np.uint8)
    mask1[setyx] = maxval

    # Estimate mask 2 from thresholding
    mask2 = cv2.threshold(image, thresh, maxval, thresh_type)[1]

    # Get binary image from estimated masks
    image = cv2.bitwise_and(mask1, mask2)

    return image


def find_contours(image, min_area=0, sort=True):
    """Find contours in image"""
    image = image.copy()

    # Find contours depending on OpenCV version
    if opencv_version() == 3:
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
    else:
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    # Filter contours by area
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    if len(contours) < 1:
        return ([], [])

    # Sort contours from left to right using respective bounding boxes
    if sort:
        bndboxes = [cv2.boundingRect(contour) for contour in contours]
        contours, bndboxes = zip(*sorted(zip(contours, bndboxes), key=lambda x: x[1][0]))

    return contours, bndboxes


def grab_region(image, bwmask, contours, bndboxes, index):
    """Extract region of interest"""
    region = np.zeros_like(bwmask, np.uint8)
    if len(contours) > 0 and len(bndboxes) > 0 and index >= 0:
        x, y, w, h = bndboxes[index]
        region = cv2.drawContours(region, contours, index, (255, 255, 255), -1, cv2.LINE_AA)
        region = region[y : y + h, x : x + w]
        bwmask = bwmask[y : y + h, x : x + w]
        bwmask = cv2.bitwise_and(region, region, mask=bwmask)
        region = image[y : y + h, x : x + w]
        region = cv2.bitwise_and(region, region, mask=bwmask)
    return region


def grab_regions(image, image_mask, contours, bndboxes):
    """Extract all regions of interest"""
    regions = []
    for index in range(len(bndboxes)):
        regions.append(grab_region(image, image_mask, contours, bndboxes, index))
    return regions


def image2tensor(image, shape, padding=0.0, rescale=1.0, color_mode=None):
    """Convert image to tensor for model input"""
    output = cv2.cvtColor(image, color_mode) if color_mode else image.copy()
    output = np.atleast_3d(output)
    rect_w = output.shape[1]
    rect_h = output.shape[0]
    sqrlen = int(np.ceil((1.0 + padding) * max(rect_w, rect_h)))
    sqrbox = np.zeros((sqrlen, sqrlen, output.shape[2]), np.uint8)
    rect_x = (sqrlen - rect_w) // 2
    rect_y = (sqrlen - rect_h) // 2
    sqrbox[rect_y : rect_y + rect_h, rect_x : rect_x + rect_w] = output
    output = cv2.resize(sqrbox, shape[:2])
    output = np.atleast_3d(output)
    output = np.asarray(output, np.float32) * rescale
    output = output.reshape((1,) + output.shape)
    return output


def char2onehot(character, alphabet):
    """Convert character to one-hot encoding"""
    onehot = [0.0] * len(alphabet)
    onehot[alphabet.index(character)] = 1.0
    onehot = np.asarray(onehot, np.float32).reshape(1, len(alphabet), 1)
    return onehot


def resize(image, w=-1, h=-1, bbox=False):
    """Resize image"""
    image = Image.fromarray(image)
    bnbox = image.getbbox() if bbox else None
    image = image.crop(bnbox) if bnbox else image
    if w <= 0 and h <= 0:
        w = image.width
        h = image.height
    elif w <= 0 and h > 0:
        w = int(image.width / image.height * h)
    elif w > 0 and h <= 0:
        h = int(image.height / image.width * w)
    else:
        pass
    image = image.resize((w, h))
    image = np.asarray(image, np.uint8)
    return image


def transfer_color_max(source, target):
    """Transfer color using maximum occurrence method"""
    colors = source.convert("RGB").getcolors(256 * 256 * 256)
    colors = sorted(colors, key=lambda x: x[0], reverse=True)
    maxcol = colors[0][1] if len(colors) == 1 else colors[0][1] if colors[0][1] != (0, 0, 0) else colors[1][1]
    output = Image.new("RGB", target.size)
    colors = Image.new("RGB", target.size, maxcol)
    output.paste(colors, (0, 0), target.convert("L"))
    return output


def update_bndboxes(bndboxes, index, image):
    """Update bounding boxes after character replacement"""
    change_x = (image.shape[1] - bndboxes[index][2]) // 2
    bndboxes = list(bndboxes)
    for i in range(0, index + 1):
        x, y, w, h = bndboxes[i]
        bndboxes[i] = (x - change_x, y, w, h)
    for i in range(index + 1, len(bndboxes)):
        x, y, w, h = bndboxes[i]
        bndboxes[i] = (x + change_x, y, w, h)
    bndboxes = tuple(bndboxes)
    return bndboxes


def paste_images(image, patches, bndboxes):
    """Paste image patches onto base image"""
    image = Image.fromarray(image)
    for patch, bndbox in zip(patches, bndboxes):
        patch = Image.fromarray(patch)
        image.paste(patch, bndbox[:2])
    image = np.asarray(image, np.uint8)
    return image


def inpaint(image, mask):
    """Inpaint image using mask"""
    k = np.ones((5, 5), np.uint8)
    m = cv2.dilate(mask, k, iterations=1)
    i = cv2.inpaint(image, m, 10, cv2.INPAINT_TELEA)
    return i


def edit_char(image, image_mask, contours, bndboxes, index, char, alphabet, fannet, colornet):
    """Edit character in image"""
    # Validate parameters
    if len(contours) <= 0 or len(bndboxes) <= 0 or len(contours) != len(bndboxes) or index < 0:
        return None, None

    # Generate character
    region_f = grab_region(image_mask, image_mask, contours, bndboxes, index)
    tensor_f = image2tensor(region_f, fannet.input_shape[0][1:3], 0.1, 1.0)
    onehot_f = char2onehot(char, alphabet)
    output_f = fannet.predict([tensor_f, onehot_f])
    output_f = np.squeeze(output_f)
    output_f = np.asarray(output_f, np.uint8)

    # Transfer color
    region_c = grab_region(image, image_mask, contours, bndboxes, index)
    source_c = Image.fromarray(region_c)
    target_f = Image.fromarray(output_f)
    output_c = transfer_color_max(source_c, target_f)
    output_c = np.asarray(output_c, np.uint8)

    output_f = resize(output_f, -1, region_f.shape[0], True)
    output_c = resize(output_c, -1, region_c.shape[0], True)

    # Inpaint old layout
    mpatches = grab_regions(image_mask, image_mask, contours, bndboxes)
    o_layout = np.zeros_like(image_mask, np.uint8)
    o_layout = paste_images(o_layout, mpatches, bndboxes)
    inpainted_image = inpaint(image, o_layout)

    # Create new layout
    bpatches = grab_regions(image, image_mask, contours, bndboxes)
    bndboxes = update_bndboxes(bndboxes, index, output_f)
    bpatches[index] = output_c
    n_layout = np.zeros_like(image, np.uint8)
    n_layout = paste_images(n_layout, bpatches, bndboxes)
    mpatches[index] = output_f
    m_layout = np.zeros_like(image_mask, np.uint8)
    m_layout = paste_images(m_layout, mpatches, bndboxes)

    # Generate final result
    n_layout = Image.fromarray(n_layout)
    m_layout = Image.fromarray(m_layout)
    inpainted_image = Image.fromarray(inpainted_image)
    inpainted_image.paste(n_layout, (0, 0), m_layout)

    layout = np.asarray(m_layout, np.uint8)
    edited = np.asarray(inpainted_image, np.uint8)

    return layout, edited


def process_image(image, target_text, net_f, net_c, points=None, thresh=150, min_contour_area=0):
    """
    Process an image to replace text with target_text

    Args:
        image: Input image (numpy array)
        target_text: Text to insert (only uppercase A-Z supported)
        points: List of points defining the text region (optional)
        thresh: Threshold for binarization
        min_contour_area: Minimum area for contours to be considered

    Returns:
        Edited image with replaced text
    """
    # Load models
    if net_f is None or net_c is None:
        print("Failed to load models")
        return None, 0

    # Read image
    if image is None:
        print("Failed to read image ")
        return None, 0

    # Convert target text to uppercase
    target_text = target_text.upper()

    # Process image
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_mask = binarize(image_gray, points, thresh, 255, 0)
    contours, bndboxes = find_contours(image_mask, min_contour_area)

    # Check if we have enough contours for the target text
    if len(contours) < len(target_text):
        # print(
        #     f"Not enough text regions detected. Found {len(contours)} regions, but need {len(target_text)} for the text."
        # )
        target_text = target_text[: len(contours)]

    # Edit each character sequentially
    num_changed = 0
    image_edit = image.copy()
    for i, char in enumerate(target_text):
        if i >= len(contours):
            break

        if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            try:
                image_mask, image_edit = edit_char(
                    image_edit, image_mask, contours, bndboxes, i, char, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", net_f, net_c
                )

                # Update contours and bounding boxes after each edit
                image_gray = cv2.cvtColor(image_edit, cv2.COLOR_BGR2GRAY)
                contours, bndboxes = find_contours(image_mask, min_contour_area)
                num_changed += 1
            except Exception as e:
                print(f"Error replacing character {char} at position {i}: {e}")

    return image_edit, num_changed


class STEFANNInference:
    def __init__(self, models_path="release/models"):
        self.models_path = models_path
        self.net_f, self.net_c = load_models(models_path=models_path)

    def infer(self, image, target_text):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        points = None
        thresh = 150
        min_contour_area = 0
        image_edit, num_changed = process_image(
            image, target_text, self.net_f, self.net_c, points, thresh, min_contour_area
        )
        if num_changed == 0:
            return None

        image_edit = cv2.cvtColor(image_edit, cv2.COLOR_BGR2RGB)
        return image_edit


def main():
    models_path = "release/models"
    image_path = "release/sample_images/01.jpg"
    target_text = "HELLOO"

    model = STEFANNInference(models_path=models_path)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = 330, 370, 465, 400
    crop = image[y1:y2, x1:x2]

    result = model.process(crop, target_text)
    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("STEFANN Result", image)
    # cv2.waitKey(0)
    # exit()
    # Process image
    # result = process_image(crop, target_text, models_path, thresh=150, min_contour_area=10)

    if result is not None:
        # Display result
        cv2.imshow("STEFANN Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to process image")


if __name__ == "__main__":
    main()
