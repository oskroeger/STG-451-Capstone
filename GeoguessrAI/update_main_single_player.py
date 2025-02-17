import pyautogui
import yaml
import os
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageChops

from select_regions import get_coords
from geoguessr_bot import GeoBot

# Load environment variables
load_dotenv()
client = OpenAI()

def check_game_over():
    """
    Detects if the GeoGuessr duel has ended by comparing the current screen 
    with the reference end screen.
    """
    if not os.path.exists("end_screen_reference.png"):
        print("Error: 'end_screen_reference.png' not found. Run capture script first.")
        return False

    # Capture the current bottom quarter of the screen (excluding last 100 pixels)
    screen_width, screen_height = pyautogui.size()
    top = screen_height - (screen_height // 4)  # Start of bottom quarter
    height = (screen_height // 4) - 100  # Exclude bottom 100 pixels
    region = (0, top, screen_width, height)

    current_screen = pyautogui.screenshot(region=region)
    
    # Load reference image
    reference_screen = Image.open("end_screen_reference.png")

    # Compute difference
    diff = ImageChops.difference(reference_screen, current_screen)

    # If no significant difference, we assume the game is over
    if diff.getbbox() is None:
        print("End screen detected! Game over.")
        return True
    return False

def wait_for_transition(bot):
    """
    Waits until the game transitions to the results screen.
    This is determined by detecting a significant change in the minimap area.
    After waiting 15 seconds, it checks if the game has ended before continuing.
    """
    print("Waiting for the round to transition...")

    # Take a reference screenshot of the minimap
    reference_map = pyautogui.screenshot(region=(bot.map_x, bot.map_y, bot.map_w, bot.map_h))

    while True:
        sleep(0.5)  # Check every 0.5 seconds

        # Capture a new screenshot of the minimap area
        current_map = pyautogui.screenshot(region=(bot.map_x, bot.map_y, bot.map_w, bot.map_h))

        # Compute the difference between the reference and current image
        diff = ImageChops.difference(reference_map, current_map)

        # If there's a noticeable difference, the map likely changed (transition detected)
        if diff.getbbox() is not None:
            print("Transition detected! Waiting 15 seconds before checking if game ended...")
            sleep(15)  # Wait for full transition before checking game status
            
            # **Check if the game has ended**
            if check_game_over():
                print("Game over detected! Stopping bot.")
                return True  # Signal that the game is over
            
            break  # Exit transition loop if game isn't over

    return False  # Game is still active, continue playing

def play_turn(bot: GeoBot, plot: bool = False):
    """
    Play a single turn in GeoGuessr using the local model for predictions.
    """
    # Capture a screenshot of the game
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)

    # Resize the screenshot to reduce model processing time
    resized_screenshot = screenshot.resize((screenshot.width // 4, screenshot.height // 4), Image.Resampling.LANCZOS)

    # Get the predicted location
    _, coordinates = bot.predict_local(resized_screenshot)
    if coordinates is not None:
        # Convert lat/lon to minimap coordinates
        x, y = bot.lat_lon_to_mercator_map_pixels(*coordinates)
        bot.select_map_location(x, y, plot=plot)
    else:
        print("Failed to get coordinates. Clicking the center of the minimap.")
        default_x = bot.map_x + bot.map_w // 2
        default_y = bot.map_y + bot.map_h // 2
        bot.select_map_location(default_x, default_y, plot=plot)

    # Press enter immediately after selecting location
    pyautogui.press("enter")
    
    # **NEW: Wait 2 seconds before checking for transition**
    sleep(2)

    # Press enter again to get rid of chat
    pyautogui.press("enter")

    # Wait for the transition and check if game is over
    game_over = wait_for_transition(bot)
    
    return game_over  # Return whether the game ended

def main(plot=False):
    """
    Main function to run the GeoGuessr bot indefinitely until the duel ends.
    """
    if "screen_regions.yaml" not in os.listdir():
        screen_regions = get_coords(players=1)
    else:
        with open("screen_regions.yaml") as f:
            screen_regions = yaml.safe_load(f)

    bot = GeoBot(screen_regions, player=1)
    
    turn = 0
    while True:  # Run indefinitely until the game ends
        turn += 1
        print(f"\nTurn {turn}")
        
        game_over = play_turn(bot, plot=plot)
        
        if game_over:
            break  # Stop playing if the game has ended

if __name__ == "__main__":
    main(plot=True)
