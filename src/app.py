import os
import gradio as gr
import threading
import time
from PIL import Image
from modular import OmniParser

# Initialize the OmniParser
parser = OmniParser()

screenshots_folder = "imgs"
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)

def take_screenshot_and_process(target_text):
    """
    Take a screenshot and process it asynchronously.
    Returns the path to the screenshot immediately.
    Starts processing in the background.
    """
    # Take a screenshot and save it
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    image_path = parser.take_screenshot(filename=filename, folder=screenshots_folder, delay=3)
    
    # Start processing in a separate thread
    def process_in_background():
        try:
            # Process the image to find elements
            parsed_content_list, _, _ = parser.process_image(image_path)
            
            # Try to click on the specified element
            success = parser.click_element_by_content(parsed_content_list, target_text)
            
            if not success:
                print(f"Failed to find and click on element: '{target_text}'")
        except Exception as e:
            print(f"Error in background processing: {str(e)}")
    
    # Start the background thread
    thread = threading.Thread(target=process_in_background)
    thread.daemon = True
    thread.start()
    
    return image_path

def on_submit(target_text):
    """
    Handler for submit button. Takes and returns screenshot immediately,
    while processing continues in the background.
    """
    if not target_text.strip():
        return None, "Please enter a target text element."
    
    try:
        # Take the screenshot and start processing
        image_path = take_screenshot_and_process(target_text)
        
        return image_path, f"Screenshot taken. Looking for '{target_text}' element..."
    except Exception as e:
        return None, f"Error: {str(e)}"


with gr.Blocks(title="OmniParser UI Automation") as app:
    gr.Markdown("# OmniParser UI Automation")
    gr.Markdown("Enter the text of an element to find and click on it.")
    
    with gr.Row():
        # Left column - Input
        with gr.Column(scale=1):
            target_input = gr.Textbox(
                label="Target Element Text",
                placeholder="Enter text like 'Video settings'...",
                lines=1
            )
            submit_btn = gr.Button("Take Screenshot & Process", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False)
        
        # Right column - Screenshot display
        with gr.Column(scale=2):
            screenshot_display = gr.Image(
                label="Screenshot",
                type="filepath",
                height=600
            )
    
    # Connect the submit button to the handler function
    submit_btn.click(
        fn=on_submit,
        inputs=[target_input],
        outputs=[screenshot_display, status_text]
    )

if __name__ == "__main__":
    app.launch(share=False)