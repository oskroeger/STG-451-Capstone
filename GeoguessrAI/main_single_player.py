import pyautogui
import yaml
import os
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from select_regions import get_coords
from geoguessr_bot import GeoBot

# Load environment variables
load_dotenv()
client = OpenAI()

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

    # Proceed to the next round
    pyautogui.press(" ")
    sleep(2)

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