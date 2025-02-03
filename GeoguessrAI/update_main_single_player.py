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

def wait_for_transition(bot):
    """
    Waits until the game transitions to the results screen.
    This is determined by detecting a significant change in the minimap area.
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
            print("Transition detected! Waiting 15 seconds before continuing...")
            sleep(15)  # Additional wait after transition
            break

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

    # Wait for the transition to the next screen
    wait_for_transition(bot)

def main(turns=5, plot=False):
    """
    Main function to run the GeoGuessr bot for multiple turns.
    """
    if "screen_regions.yaml" not in os.listdir():
        screen_regions = get_coords(players=1)
    else:
        with open("screen_regions.yaml") as f:
            screen_regions = yaml.safe_load(f)

    bot = GeoBot(screen_regions, player=1)
    for turn in range(turns):
        print(f"\nTurn {turn + 1}/{turns}")
        play_turn(bot, plot=plot)

if __name__ == "__main__":
    main(turns=5, plot=True)
