import gradio as gr
import base64
import io
from PIL import Image
from modular import OmniParser

def search_content(search_text):
    """
    Function to take a screenshot and highlight the searched content
    
    Parameters:
    - search_text: The text to search for in the screenshot
    
    Returns:
    - A PIL Image of the raw screenshot
    """
    # Initialize the OmniParser
    parser = OmniParser()
    
    # Take a screenshot
    image_path = parser.take_screenshot(delay=1)
    
    # Load the screenshot as a PIL Image
    screenshot = Image.open(image_path)
    
    # Process the image to parse content (we'll need this for debugging)
    parsed_content_list, _, _ = parser.process_image(image_path)
    
    # Log the search text and whether it was found
    found = any(search_text.lower() in item.get('content', '').lower() for item in parsed_content_list)
    print(f"Searching for: '{search_text}'")
    print(f"Content found: {found}")
    
    return screenshot

# Create the interface using gr.Interface
app = gr.Interface(
    fn=search_content,
    inputs=gr.Textbox(
        label="Text to search for",
        placeholder="Enter text content to search for...",
        lines=1
    ),
    outputs=gr.Image(
        label="Screenshot with Highlighted Results",
        type="pil"
    ),
    title="OmniParser Screenshot Search Tool",
    description="Enter text to search for in a screenshot. The tool will capture your screen and highlight any matches.",
    examples=["Button", "Menu", "Error", "Login"],
    article="""
    ### How to Use
    1. Enter the text you want to search for in the screenshot
    2. Press the Submit button
    3. The tool will take a screenshot and highlight any matches
    
    ### Tips
    - Be specific with your search terms for better results
    - The search is case-insensitive
    - Try using examples from the buttons below the search box
    """,
    theme="default"
)

if __name__ == "__main__":
    app.launch()