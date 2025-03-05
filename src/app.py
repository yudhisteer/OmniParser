import os
import gradio as gr
import threading
import time
import logging
from PIL import Image
from modular import OmniParser

# Initialize the OmniParser
parser = OmniParser()

# Create a folder for screenshots if it doesn't exist
screenshots_folder = "screenshots"
if not os.path.exists(screenshots_folder):
    os.makedirs(screenshots_folder)

# Global variables
log_messages = []
processing_active = False
last_processed_log_index = 0

# Original logger
original_info = parser.logger.info

# Override the logger's info method to capture logs
def custom_info(msg, *args, **kwargs):
    global log_messages
    
    # Call the original method
    original_info(msg, *args, **kwargs)
    
    # Store messages we're interested in
    if any(keyword in msg for keyword in ['Taking screenshot', 'Moving mouse', 'click performed', 'found midpoint']):
        log_messages.append(msg)

# Replace the logger's info method with our custom one
parser.logger.info = custom_info

def process_and_update(element_text, chat_history):
    """
    Process the screenshot and find the specified element,
    then continuously update the chat with log messages.
    """
    global log_messages, processing_active, last_processed_log_index
    
    # Reset log messages and set processing as active
    log_messages = []
    processing_active = True
    last_processed_log_index = 0
    
    # Add user input to chat history
    chat_history = chat_history + [(element_text, None)]
    yield chat_history, None
    
    # Initial response
    initial_response = f"Looking for element: '{element_text}'..."
    chat_history = chat_history[:-1] + [(element_text, initial_response)]
    yield chat_history, None
    
    try:
        # Take screenshot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        image_path = parser.take_screenshot(filename=filename, folder=screenshots_folder)
        
        # Update with the screenshot
        yield chat_history, image_path
        
        # Process the image in a separate thread
        def background_process():
            global processing_active
            try:
                # Process the image
                parsed_content, _, _ = parser.process_image(image_path)
                
                # Try to click on the element
                success = parser.click_element_by_content(parsed_content, element_text)
                
                # Wait a bit more for any final logs
                time.sleep(2)
                
                # Mark processing as complete
                processing_active = False
                
            except Exception as e:
                print(f"Error in background processing: {str(e)}")
                processing_active = False
        
        # Start the background thread
        bg_thread = threading.Thread(target=background_process)
        bg_thread.daemon = True
        bg_thread.start()
        
        # Update the chat with screenshot taken message
        response = initial_response + f"\n\nScreenshot taken: {filename}"
        chat_history = chat_history[:-1] + [(element_text, response)]
        yield chat_history, image_path
        
        # Continue updating until processing is complete or timeout
        last_response = response
        max_wait_time = 30  # Maximum time to wait for processing in seconds
        start_time = time.time()
        
        while processing_active and (time.time() - start_time) < max_wait_time:
            # Check if there are new log messages
            if last_processed_log_index < len(log_messages):
                # Get new logs
                new_logs = log_messages[last_processed_log_index:]
                last_processed_log_index = len(log_messages)
                
                # Update the response with new logs
                last_response = last_response + "\n\n" + "\n".join(new_logs)
                chat_history = chat_history[:-1] + [(element_text, last_response)]
                yield chat_history, image_path
            
            # Wait a short time before checking again
            time.sleep(0.5)
        
        # Final update after processing completes
        if last_processed_log_index < len(log_messages):
            new_logs = log_messages[last_processed_log_index:]
            last_response = last_response + "\n\n" + "\n".join(new_logs)
        
        # Add completion message
        if "click performed" in last_response:
            last_response += "\n\n✅ Task completed successfully!"
        else:
            last_response += "\n\n⚠️ Processing completed but the element may not have been found."
        
        chat_history = chat_history[:-1] + [(element_text, last_response)]
        return chat_history, image_path
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history = chat_history[:-1] + [(element_text, error_msg)]
        processing_active = False
        return chat_history, None

# Create the Gradio interface
with gr.Blocks(title="OmniParser Chat") as app:
    gr.Markdown("# OmniParser Chat")
    gr.Markdown("Enter the text of an element to find and click (like 'Video settings').")
    
    with gr.Row():
        # Left column - Chat
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(height=500)
            
            with gr.Row():
                with gr.Column(scale=8):
                    txt = gr.Textbox(
                        show_label=False, 
                        placeholder="Enter element text...",
                        container=False
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Send")
            
            with gr.Row():
                clear_btn = gr.Button("Clear")
        
        # Right column - Screenshot
        with gr.Column(scale=2):
            image_output = gr.Image(type="filepath", label="Screenshot")
    
    # Define the event handlers
    submit_btn.click(
        fn=process_and_update,
        inputs=[txt, chatbot],
        outputs=[chatbot, image_output]
    ).then(lambda: "", None, txt)  # Clear the textbox
    
    txt.submit(
        fn=process_and_update,
        inputs=[txt, chatbot],
        outputs=[chatbot, image_output]
    ).then(lambda: "", None, txt)  # Clear the textbox
    
    clear_btn.click(lambda: [], None, chatbot)
    clear_btn.click(lambda: None, None, image_output)

# Launch the app
if __name__ == "__main__":
    app.launch(share=False)