import io
import base64
from PIL import Image


def image_to_base64(img: Image) -> str:
    # Create a byte stream to hold the image data
    buffered = io.BytesIO()
    
    # Save the image to the byte stream in a specific format (e.g., PNG)
    img.save(buffered, format="PNG")
    
    # Get the byte data from the byte stream
    img_bytes = buffered.getvalue()
    
    # Convert the byte data to a Base64 string
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    
    return img_base64
