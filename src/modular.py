import base64
import datetime
import io
import logging
import os
import time
import sys
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import pyautogui
import torch
from PIL import Image

from util.utils import (check_ocr_box, get_caption_model_processor,
                        get_som_labeled_img, get_yolo_model)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o-mini"




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
    
    
    def _get_button_location(self, client, model, parsed_content_list, target_content):
        """
        Extract the bounding box of a button using OpenAI's parse functionality.
        
        Parameters:
        - client: OpenAI client instance
        - model: The model to use for parsing
        - parsed_content_list: List of parsed content items from OCR
        - target_content: The button text to search for
        
        Returns:
        - ButtonLocation object with bbox or None if not found
        """
        class ButtonLocation(BaseModel):
            bbox: list[float] = Field(description="Bounding box of button [x1, y1, x2, y2]")
            
        try:
            completion = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": f"Extract the bounding box of the button from {parsed_content_list}. Find the closest match to '{target_content}' which makes sense. Return the bbox as [x1, y1, x2, y2]."},
                    {"role": "user", "content": target_content},
                ],
                response_format=ButtonLocation,
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            self.logger.error(f"Error extracting button location: {e}")
            return None
        
    def _calculate_midpoint(self, bbox):
        """
        Calculate the midpoint of a bounding box.
        
        Parameters:
        - bbox: List [x1, y1, x2, y2] representing the bounding box
        
        Returns:
        - (x, y): Tuple containing the midpoint coordinates
        """
        if not bbox or len(bbox) != 4:
            return None
        
        midpoint_x = (bbox[0] + bbox[2]) / 2
        midpoint_y = (bbox[1] + bbox[3]) / 2
        return (midpoint_x, midpoint_y)
        

    def get_midpoint_by_content(self, parsed_content_list, target_content):
        """
        Find the midpoint of a bounding box for a specific button.
        
        Parameters:
        - client: OpenAI client instance
        - model: The model to use for parsing
        - parsed_content_list: List of parsed content items from OCR
        - target_content: The button text to search for
        
        Returns:
        - (x, y): Tuple containing the midpoint coordinates or None if not found
        """
        button_location = self._get_button_location(client, model, parsed_content_list, target_content)
        
        if button_location and hasattr(button_location, 'bbox'):
            midpoint = self._calculate_midpoint(button_location.bbox)
            self.logger.info(f"Found midpoint for '{target_content}': {midpoint}")
            return midpoint
        
        self.logger.warning(f"Button '{target_content}' not found in parsed content list")

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
    # parsed_content, image_path = parser.analyze_screenshot(delay=3, display=False)
    # print("Parsed content: ", parsed_content, "\n")
    parsed_content = [{'type': 'text', 'bbox': [0.046875, 0.009027778171002865, 0.076171875, 0.02916666679084301], 'interactivity': False, 'content': 'Camera', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.03242187574505806, 0.4722222089767456, 0.05820312350988388, 0.49444442987442017], 'interactivity': False, 'content': 'About', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.701953113079071, 0.4888888895511627, 0.736328125, 0.5145833492279053], 'interactivity': False, 'content': 'Preview', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.06562499701976776, 0.5201388597488403, 0.09882812201976776, 0.5402777791023254], 'interactivity': False, 'content': 'Camera', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.614453136920929, 0.5256944298744202, 0.668749988079071, 0.5513888597488403], 'interactivity': False, 'content': '2025.2501.1.0', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.064453125, 0.5347222089767456, 0.1875, 0.5590277910232544], 'interactivity': False, 'content': ' 2025 Microsoft. All rights reserved.', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.03125, 0.5826388597488403, 0.08945312350988388, 0.6083333492279053], 'interactivity': False, 'content': ' Send feedback', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.03242187574505806, 0.6305555701255798, 0.05429687350988388, 0.6527777910232544], 'interactivity': False, 'content': 'Help', 'source': 'box_ocr_content_ocr'}, {'type': 'text', 'bbox': [0.02382812462747097, 0.9729166626930237, 0.076171875, 0.9944444298744202], 'interactivity': False, 'content': 'Mostly cloudy', 'source': 'box_ocr_content_ocr'}, {'type': 'icon', 'bbox': [0.03231000900268555, 0.20929764211177826, 0.6964194178581238, 0.28781506419181824], 'interactivity': True, 'content': 'Photo settings ', 'source': 'box_yolo_content_ocr'}, {'type': 'icon', 'bbox': [0.029300928115844727, 0.36193007230758667, 0.6936324834823608, 0.43950068950653076], 'interactivity': True, 'content': 'Related settings ', 'source': 'box_yolo_content_ocr'}, {'type': 'icon', 'bbox': [0.03097982332110405, 0.1340823471546173, 0.7019819021224976, 0.21152812242507935], 'interactivity': True, 'content': 'Camera settings ', 'source': 'box_yolo_content_ocr'}, {'type': 'icon', 'bbox': [0.9581998586654663, 0.9503650069236755, 0.9949334859848022, 0.9943264722824097], 'interactivity': True, 'content': '8:52 PM 3/4/2025 ', 'source': 'box_yolo_content_ocr'}, {'type': 'icon', 'bbox': [0.06277003139257431, 0.2874622642993927, 0.6923016905784607, 0.36506956815719604], 'interactivity': True, 'content': 'Video settings ', 'source': 'box_yolo_content_ocr'}, {'type': 'icon', 'bbox': [0.031483445316553116, 0.05848612263798714, 0.12059949338436127, 0.10878979414701462], 'interactivity': True, 'content': 'Settings ', 'source': 'box_yolo_content_ocr'}, {'type': 'icon', 'bbox': [0.702910304069519, 0.136135995388031, 0.96973717212677, 0.4899347126483917], 'interactivity': True, 'content': 'a person working.', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.5928689241409302, 0.9535789489746094, 0.617752730846405, 0.9974761009216309], 'interactivity': True, 'content': 'Google Chrome web browser', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.6184104084968567, 0.9540834426879883, 0.6428107023239136, 0.9973570108413696], 'interactivity': True, 'content': 'Microsoft 365', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.56634920835495, 0.9525105953216553, 0.5908659100532532, 0.9994043707847595], 'interactivity': True, 'content': 'Toggle Terminal', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.3314294219017029, 0.9555178880691528, 0.46262669563293457, 0.9926130771636963], 'interactivity': True, 'content': 'a video-related function.', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.3068317472934723, 0.953494668006897, 0.3297480642795563, 0.9931134581565857], 'interactivity': True, 'content': 'Windows', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.5390022397041321, 0.9521636962890625, 0.5659992098808289, 1.0], 'interactivity': True, 'content': 'Gallery', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.5152496099472046, 0.9520431756973267, 0.5393550992012024, 1.0], 'interactivity': True, 'content': 'Strikethrough', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.4894563555717468, 0.9508355259895325, 0.51482093334198, 1.0], 'interactivity': True, 'content': 'OneNote', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.6691263318061829, 0.9530095458030701, 0.6950151324272156, 1.0], 'interactivity': True, 'content': 'Oval Viewer', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.6435226202011108, 0.952664852142334, 0.6691875457763672, 0.9999040365219116], 'interactivity': True, 'content': 'View', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.4638824462890625, 0.950718343257904, 0.48935452103614807, 0.9978310465812683], 'interactivity': True, 'content': 'Copy', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.9204201698303223, 0.0, 0.9444247484207153, 0.03890472650527954], 'interactivity': True, 'content': 'Minimize', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.908613383769989, 0.9546249508857727, 0.9249629974365234, 0.9891508221626282], 'interactivity': True, 'content': 'WiFi', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.9248651266098022, 0.9534440636634827, 0.95196133852005, 0.9894493818283081], 'interactivity': True, 'content': 'Sound', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.8886600732803345, 0.9549524188041687, 0.9086421728134155, 0.990831732749939], 'interactivity': True, 'content': 'Refresh', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.004158693365752697, 0.0, 0.023795777931809425, 0.03347144275903702], 'interactivity': True, 'content': 'Back', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.8716408014297485, 0.9558804035186768, 0.8887779116630554, 0.9902165532112122], 'interactivity': True, 'content': 'Move Up', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.9465262293815613, 0.0, 0.9713810682296753, 0.0369967520236969], 'interactivity': True, 'content': 'Maximize', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.025004934519529343, 0.0, 0.044040076434612274, 0.032085951417684555], 'interactivity': True, 'content': 'Image', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.9739215970039368, 0.0, 0.9972831606864929, 0.037009477615356445], 'interactivity': True, 'content': 'Close', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.9942995309829712, 0.9505579471588135, 1.0, 0.9897557497024536], 'interactivity': True, 'content': 'a stop button.', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [2.6226043701171875e-06, 0.9504732489585876, 0.005570451728999615, 0.9967041611671448], 'interactivity': True, 'content': 'a stop button.', 'source': 'box_yolo_content_yolo'}, {'type': 'icon', 'bbox': [0.005290811415761709, 0.9500875473022461, 0.029621237888932228, 0.9983165264129639], 'interactivity': True, 'content': 'Notifications', 'source': 'box_yolo_content_yolo'}]

    for idx, item in enumerate(parsed_content):
        print(f"ID: {idx}, Type: {item['type']}, Content: {item['content']}")


    # Find the midpoint of the "Settings" element
    BUTTON = "Back"
    midpoint = parser.get_midpoint_by_content(parsed_content, BUTTON)
    if midpoint:
        x, y = midpoint
        print(f"{BUTTON} midpoint: x={x}, y={y}")


    # # Click on the "Settings" element
    # parser.click_element_by_content(parsed_content, BUTTON)
