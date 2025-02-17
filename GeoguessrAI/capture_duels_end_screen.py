import pyautogui
from PIL import Image

def capture_end_screen():
    """
    Captures the bottom quarter of the screen, excluding the last 100 pixels, 
    and saves it as a reference image.
    """
    screen_width, screen_height = pyautogui.size()
    
    # Define the region: bottom quarter, but 100 pixels up excluded
    top = screen_height - (screen_height // 4)  # Bottom quarter start
    height = (screen_height // 4) - 100  # Remove 100 pixels from the bottom

    region = (0, top, screen_width, height)
    
    # Take a screenshot of the defined region
    screenshot = pyautogui.screenshot(region=region)
    
    # Save the screenshot
    screenshot.save("end_screen_reference.png")
    print("End screen reference saved as 'end_screen_reference.png'")

if __name__ == "__main__":
    capture_end_screen()
