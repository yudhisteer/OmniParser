import base64
import datetime
import io
import logging
import os
import time
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pyautogui
import torch
from PIL import Image

from util.utils import (check_ocr_box, get_caption_model_processor,
                        get_som_labeled_img, get_yolo_model)


class OmniParser:
    """
    OmniParser class for screen parsing and analysis with OCR and object detection.
    Provides functionality to capture screenshots and analyze their content.
    """
    
    def __init__(self, yolo_model_path="weights/icon_detect/model.pt", 
                 caption_model_name="florence2", 
                 caption_model_path="weights/icon_caption_florence"):
        """
        Initialize the OmniParser with models and configurations.
        
        Parameters:
        - yolo_model_path: Path to the YOLO model weights
        - caption_model_name: Name of the caption model
        - caption_model_path: Path to the caption model
        """
        # Initialize logger
        self.logger = self._setup_logger()
        self.logger.info("Starting OmniParser initialization...")
        
        # Check CUDA availability
        self._check_cuda_availability()
        
        # Suppress transformers warnings
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        import transformers
        transformers.logging.set_verbosity_error()
        
        # Initialize device and models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_models(yolo_model_path, caption_model_name, caption_model_path)
        
        # Configuration for box overlay
        self.box_treshold = 0.05
        
    def _setup_logger(self):
        """Set up and configure the logger."""
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
        
        return logger
    
    def _check_cuda_availability(self):
        """Check and log CUDA availability and details."""
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    def _initialize_models(self, yolo_model_path, caption_model_name, caption_model_path):
        """Initialize the YOLO and caption models."""
        self.logger.info("Loading YOLO model...")
        self.yolo_model = get_yolo_model(model_path=yolo_model_path)
        
        self.logger.info("Loading caption model processor...")
        self.caption_model_processor = get_caption_model_processor(
            model_name=caption_model_name, 
            model_name_or_path=caption_model_path
        )
        
        # Move models to device
        self.yolo_model = self.yolo_model.to(self.device)
        if hasattr(self.caption_model_processor, "model"):
            self.caption_model_processor.model = self.caption_model_processor.model.to(self.device)
        
        self.logger.info("Models loaded successfully!")
        self.logger.info("OmniParser initialization complete!")
    
    def take_screenshot(self, filename=None, folder="imgs", delay=0):
        """
        Take a screenshot and save it to the specified folder.
        
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
            self.logger.info(f"Created directory: {folder}")
        
        # Wait if delay is specified
        if delay > 0:
            self.logger.info(f"Taking screenshot in {delay} seconds...")
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
        self.logger.info(f"Screenshot saved to: {filepath}")
        
        return filepath
    
    def process_image(self, image_path):
        """
        Process an image with OCR and YOLO object detection.
        
        Parameters:
        - image_path: Path to the image to process
        
        Returns:
        - parsed_content_list: List of parsed content from the image
        - dino_labled_img: Base64 encoded image with bounding boxes
        - image: PIL Image object of the loaded image
        """
        # Load image
        image = Image.open(image_path)
        image_rgb = image.convert("RGB")
        self.logger.info(f"Image loaded successfully! with size {image_rgb.size}")

        # Configure box overlay
        box_overlay_ratio = max(image.size) / 3200
        draw_bbox_config = {
            "text_scale": 0.8 * box_overlay_ratio,
            "text_thickness": max(int(2 * box_overlay_ratio), 1),
            "text_padding": max(int(3 * box_overlay_ratio), 1),
            "thickness": max(int(3 * box_overlay_ratio), 1),
        }

        # OCR Processing
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
        self.logger.info(f"OCR time: {cur_time_ocr - start} seconds")

        # YOLO Processing
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_path,
            self.yolo_model,
            BOX_TRESHOLD=self.box_treshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128,
        )
        cur_time_caption = time.time()
        self.logger.info(f"Caption time: {cur_time_caption - cur_time_ocr} seconds")
        # self.logger.info(f"Parsed content list: {parsed_content_list}")
        
        return parsed_content_list, dino_labled_img, image
    
    def display_results(self, dino_labled_img):
        """
        Display the processed image with bounding boxes.
        
        Parameters:
        - dino_labled_img: Base64 encoded image with bounding boxes
        """
        plt.figure(figsize=(15, 15))
        image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        plt.axis("off")
        plt.imshow(image)
        plt.show()
    
    def analyze_screenshot(self, filename=None, folder="imgs", delay=0, display=True):
        """
        Take a screenshot and analyze it with OCR and YOLO.
        
        Parameters:
        - filename: Name of the file (without extension). If None, uses timestamp.
        - folder: Folder to save the image to (default: 'imgs')
        - delay: Seconds to wait before taking the screenshot (default: 0)
        - display: Whether to display the results (default: True)
        
        Returns:
        - parsed_content_list: List of parsed content from the image
        - image_path: Path to the saved screenshot
        """
        # Take screenshot
        image_path = self.take_screenshot(filename, folder, delay)
        
        # Process the image
        parsed_content_list, dino_labled_img, _ = self.process_image(image_path)
        
        # Display results if requested
        if display:
            self.display_results(dino_labled_img)
        
        return parsed_content_list, image_path
    
    def _normalize_text(self, text):
        """
        Normalize text for consistent matching by removing leading/trailing whitespace 
        and converting to lowercase.
        
        Parameters:
        - text: The text string to normalize
        
        Returns:
        - Normalized text string
        """
        if text is None:
            return ""
        
        # Strip whitespace and convert to lowercase
        normalized = text.strip().lower()
    
        return normalized
    
    def get_midpoint_by_content(self, parsed_content_list, target_content):
        """
        Find the midpoint of a bounding box for a specific content item.
        
        Parameters:
        - parsed_content_list: List of parsed content items from process_image or analyze_screenshot
        - target_content: The content text to search for
        
        Returns:
        - (x, y): Tuple containing the midpoint coordinates of the bounding box
                 or None if content not found
        """
        normalized_target_content = self._normalize_text(target_content)
        for item in parsed_content_list:
            normalized_item_content = self._normalize_text(item.get('content'))
            if normalized_item_content == normalized_target_content:
                bbox = item.get('bbox')
                if bbox and len(bbox) == 4:
                    # Calculate midpoint of the bounding box [x1, y1, x2, y2]
                    midpoint_x = (bbox[0] + bbox[2]) / 2
                    midpoint_y = (bbox[1] + bbox[3]) / 2
                    self.logger.info(f"Found midpoint for '{target_content}': ({midpoint_x}, {midpoint_y})")
                    return midpoint_x, midpoint_y
        
        self.logger.warning(f"Content '{target_content}' not found in parsed content list")
        return None
    

    def click_element_by_content(self, parsed_content_list, target_content, click_type='single', 
                                move_duration=0.5, random_offset=5, pre_click_delay=0.2, post_click_delay=0.5):
        """
        Click on a UI element based on its content text.
        
        Parameters:
        - parsed_content_list: List of parsed content items from process_image or analyze_screenshot
        - target_content: The content text to search for
        - click_type: Type of click - 'single', 'double', or 'right' (default: 'single')
        - move_duration: Time taken for mouse movement in seconds (default: 0.5)
        - random_offset: Random pixel offset to add to click position for more natural clicks (default: 5)
        - pre_click_delay: Delay before clicking in seconds (default: 0.2)
        - post_click_delay: Delay after clicking in seconds (default: 0.5)
        
        Returns:
        - bool: True if element was found and clicked, False otherwise
        """
        # Get the midpoint coordinates
        midpoint = self.get_midpoint_by_content(parsed_content_list, target_content)
        if not midpoint:
            return False
        
        x_rel, y_rel = midpoint
        
        # Get screen dimensions
        screen_width, screen_height = pyautogui.size()
        
        # Convert relative coordinates to absolute screen coordinates
        x_abs = int(x_rel * screen_width)
        y_abs = int(y_rel * screen_height)
        
        # Add small random offset for more natural clicking if specified
        if random_offset > 0:
            import random
            x_abs += random.randint(-random_offset, random_offset)
            y_abs += random.randint(-random_offset, random_offset)
        
        # Move mouse to target
        self.logger.info(f"Moving mouse to: ({x_abs}, {y_abs}) to click on '{target_content}'")
        pyautogui.moveTo(x_abs, y_abs, duration=move_duration)
        
        # Optional pre-click delay
        if pre_click_delay > 0:
            time.sleep(pre_click_delay)
        
        # Perform the requested click type
        if click_type == 'single':
            pyautogui.click()
            self.logger.info(f"Single click performed on '{target_content}'")
        elif click_type == 'double':
            pyautogui.doubleClick()
            self.logger.info(f"Double click performed on '{target_content}'")
        elif click_type == 'right':
            pyautogui.rightClick()
            self.logger.info(f"Right click performed on '{target_content}'")
        else:
            self.logger.warning(f"Unknown click type: {click_type}")
            return False
        
        # Optional post-click delay
        if post_click_delay > 0:
            time.sleep(post_click_delay)
            
        return True



if __name__ == "__main__":
    # Initialize the OmniParser
    parser = OmniParser()
    
    # Take and analyze a screenshot
    parsed_content, image_path = parser.analyze_screenshot(delay=3, display=False)
    print("Parsed content: ", parsed_content, "\n")

    # Find the midpoint of the "Settings" element
    BUTTON = "Video settings"
    midpoint = parser.get_midpoint_by_content(parsed_content, BUTTON)
    if midpoint:
        x, y = midpoint
        print(f"{BUTTON} midpoint: x={x}, y={y}")


    # Click on the "Settings" element
    parser.click_element_by_content(parsed_content, BUTTON)
