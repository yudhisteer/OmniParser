import os
import gradio as gr
import threading
from modular import OmniParser

# Initialize the OmniParser
parser = OmniParser()

# Create a folder for screenshots if it doesn't exist
screenshots_folder = "imgs"
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)

# Global flag to prevent multiple requests at once
is_processing = False

def take_screenshot_and_process(element_text, history):
    """
    Simple function to take a screenshot and process it.
    No real-time updates, just take the screenshot and return.
    """
    global is_processing
    
    # Check if we're already processing a request
    if is_processing:
        history.append((element_text, "System is busy. Please wait for the current operation to complete."))
        return history, None
    
    # Mark as processing
    is_processing = True
    
    try:
        # Add the request to history
        history.append((element_text, None))
        
        # Take the screenshot
        image_path = parser.take_screenshot(folder=screenshots_folder)
        
        # Start a separate thread to process in the background
        def process_in_background():
            global is_processing
            try:
                # Process the image
                parsed_content, _, _ = parser.process_image(image_path)
                
                # Try to click on the element
                parser.click_element_by_content(parsed_content, element_text)
            finally:
                # Always mark as not processing when done
                is_processing = False
        
        # Start the thread
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        # Return immediately with the screenshot and a simple message
        message = f"Looking for '{element_text}'. Screenshot taken and processing in background."
        history[-1] = (element_text, message)
        
        return history, image_path
        
    except Exception as e:
        # Handle errors
        error_message = f"Error: {str(e)}"
        if len(history) > 0:
            history[-1] = (element_text, error_message)
        else:
            history.append((element_text, error_message))
        
        # Reset processing flag
        is_processing = False
        return history, None

# Create a simple Gradio interface
with gr.Blocks(title="Simple OmniParser") as app:
    gr.Markdown("# Simple OmniParser")
    gr.Markdown("Enter text to find on screen. The system will take a screenshot and try to click on the element.")
    
    with gr.Row():
        # Left side - Chat and input
        with gr.Column():
            chatbot = gr.Chatbot(height=400)
            
            with gr.Row():
                text_input = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text to find...",
                    lines=1
                )
                submit_btn = gr.Button("Send")
        
        # Right side - Screenshot
        with gr.Column():
            image_output = gr.Image(type="filepath", label="Screenshot")
    
    # Clear button
    clear_btn = gr.Button("Clear")
    
    # Connect the components
    submit_btn.click(
        fn=take_screenshot_and_process,
        inputs=[text_input, chatbot],
        outputs=[chatbot, image_output]
    ).then(lambda: "", None, text_input)
    
    text_input.submit(
        fn=take_screenshot_and_process,
        inputs=[text_input, chatbot],
        outputs=[chatbot, image_output]
    ).then(lambda: "", None, text_input)
    
    clear_btn.click(lambda: [], None, chatbot)
    clear_btn.click(lambda: None, None, image_output)

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)