import base64
import datetime
import io
import logging
import os
import time

import matplotlib.pyplot as plt
import pyautogui
import torch
from PIL import Image
from ultralytics import YOLO

from util.utils import (check_ocr_box, get_caption_model_processor,
                        get_som_labeled_img, get_yolo_model)

# --------------------------------------------------------------
# Step 1: Initialize the logger
# --------------------------------------------------------------

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

# Remove any existing handlers to avoid duplicate logs
if logger.handlers:
    logger.handlers.clear()

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("\n>>> %(asctime)s [%(levelname)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Suppress the transformers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import transformers

transformers.logging.set_verbosity_error()


logger.info("Starting OmniParser initialization...")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")


def take_screenshot(filename=None, folder="imgs", delay=0):
    """
    Takes a screenshot and saves it to the specified folder.
    
    Parameters:
    - filename: Name of the file (without extension). If None, uses timestamp.
    - folder: Folder to save the image to (default: 'imgs')
    - delay: Seconds to wait before taking the screenshot (default: 0)
    
    Returns:
    - Path to the saved screenshot
    """
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    
    # Wait if delay is specified
    if delay > 0:
        print(f"Taking screenshot in {delay} seconds...")
        time.sleep(delay)
    
    # Generate filename with timestamp if not provided
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}"
    
    # Make sure filename has .png extension
    if not filename.lower().endswith('.png'):
        filename += '.png'
    
    # Complete path
    filepath = os.path.join(folder, filename)
    
    # Take the screenshot
    screenshot = pyautogui.screenshot()
    
    # Save the image
    screenshot.save(filepath)
    logger.info(f"Screenshot saved to: {filepath}")
    
    return filepath


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Step 2: Initialize the models
    # --------------------------------------------------------------

    DEVICE = torch.device("cuda")
    logger.info("Loading YOLO model...")
    yolo_model = get_yolo_model(model_path="weights/icon_detect/model.pt")
    logger.info("Loading caption model processor...")
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", model_name_or_path="weights/icon_caption_florence"
    )
    yolo_model = yolo_model.to(DEVICE)
    if hasattr(caption_model_processor, "model"):
        caption_model_processor.model = caption_model_processor.model.to(DEVICE)
    logger.info("Models loaded successfully!")
    logger.info("OmniParser initialization complete!")

    # --------------------------------------------------------------
    # Step 3: Image loading and preprocessing
    # --------------------------------------------------------------

    image_path = take_screenshot()
    image = Image.open(image_path)
    image_rgb = image.convert("RGB")
    logger.info(f"Image loaded successfully! with size {image_rgb.size}")

    box_overlay_ratio = max(image.size) / 3200
    draw_bbox_config = {
        "text_scale": 0.8 * box_overlay_ratio,
        "text_thickness": max(int(2 * box_overlay_ratio), 1),
        "text_padding": max(int(3 * box_overlay_ratio), 1),
        "thickness": max(int(3 * box_overlay_ratio), 1),
    }
    BOX_TRESHOLD = 0.05

    # --------------------------------------------------------------
    # Step 4: OCR Processing
    # --------------------------------------------------------------

    start = time.time()
    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
        image_path,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        easyocr_args={"paragraph": False, "text_threshold": 0.9},
        use_paddleocr=True,
    )
    text, ocr_bbox = ocr_bbox_rslt
    cur_time_ocr = time.time()
    logger.info(f"OCR time: {cur_time_ocr - start} seconds")

    # --------------------------------------------------------------
    # Step 5: YOLO Processing
    # --------------------------------------------------------------

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_path,
        yolo_model,
        BOX_TRESHOLD=BOX_TRESHOLD,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config,
        caption_model_processor=caption_model_processor,
        ocr_text=text,
        use_local_semantics=True,
        iou_threshold=0.7,
        scale_img=False,
        batch_size=128,
    )
    cur_time_caption = time.time()
    logger.info(f"Caption time: {cur_time_caption - cur_time_ocr} seconds")

    # --------------------------------------------------------------
    # Step 6: Displaying results
    # --------------------------------------------------------------

    logger.info(f"Parsed content list: {parsed_content_list}")

    plt.figure(figsize=(15, 15))
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    plt.axis("off")
    plt.imshow(image)
    plt.show()
