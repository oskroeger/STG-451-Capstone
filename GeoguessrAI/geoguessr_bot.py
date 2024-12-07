from io import BytesIO
import sys
import os
import torch
import re
import dotenv
import base64
import pyautogui
import matplotlib.pyplot as plt
import math
from time import sleep
from typing import Tuple, List
from PIL import Image
from openai import OpenAI

# Add LocatorBot to the Python path
locator_bot_path = os.path.abspath("../LocatorBot")
if locator_bot_path not in sys.path:
    sys.path.append(locator_bot_path)

# Import functions from LocatorBot
from train_model import load_model_and_classes, preprocess_image, predict_country

class GeoBot:
    def __init__(self, screen_regions, player=1):
        self.player = player
        self.screen_regions = screen_regions
        self.screen_x, self.screen_y = screen_regions["screen_top_left"]
        self.screen_w = screen_regions["screen_bot_right"][0] - self.screen_x
        self.screen_h = screen_regions["screen_bot_right"][1] - self.screen_y
        self.screen_xywh = (self.screen_x, self.screen_y, self.screen_w, self.screen_h)

        self.map_x, self.map_y = screen_regions[f"map_top_left_{player}"]
        self.map_w = screen_regions[f"map_bottom_right_{player}"][0] - self.map_x
        self.map_h = screen_regions[f"map_bottom_right_{player}"][1] - self.map_y
        self.minimap_xywh = (self.map_x, self.map_y, self.map_w, self.map_h)

        self.next_round_button = screen_regions["next_round_button"] if player == 1 else None
        self.confirm_button = screen_regions[f"confirm_button_{player}"]

        self.kodiak_x, self.kodiak_y = screen_regions[f"kodiak_{player}"]
        self.hobart_x, self.hobart_y = screen_regions[f"hobart_{player}"]
        self.kodiak_lat, self.kodiak_lon = (57.7916, -152.4083)
        self.hobart_lat, self.hobart_lon = (-42.8833, 147.3355)

        # Load the trained model and class names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names = load_model_and_classes(
            path="../LocatorBot/models/country_classifier.pth",
            device=self.device
        )

    def predict_local(self, screenshot: Image) -> Tuple[str, Tuple[float, float]]:
        """
        Predict the country and coordinates using the local model.
        """
        # Preprocess the screenshot
        image_tensor = preprocess_image(screenshot).to(self.device)

        # Predict using the local model
        predicted_country, coordinates = predict_country(self.model, self.class_names, image_tensor)
        print(f"Predicted Country: {predicted_country}, Coordinates: {coordinates}")
        return predicted_country, coordinates

    def lat_lon_to_mercator_map_pixels(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        Convert latitude and longitude to pixel coordinates on the minimap.
        """
        lon_diff_ref = self.kodiak_lon - self.hobart_lon
        lon_diff = self.kodiak_lon - lon
        x = abs(self.kodiak_x - self.hobart_x) * (lon_diff / lon_diff_ref) + self.kodiak_x

        mercator_y1 = math.log(math.tan(math.pi / 4 + math.radians(self.kodiak_lat) / 2))
        mercator_y2 = math.log(math.tan(math.pi / 4 + math.radians(self.hobart_lat) / 2))
        mercator_y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

        lat_diff_ref = mercator_y1 - mercator_y2
        lat_diff = mercator_y1 - mercator_y
        y = abs(self.kodiak_y - self.hobart_y) * (lat_diff / lat_diff_ref) + self.kodiak_y

        return round(x), round(y)

    def select_map_location(self, x: int, y: int, plot: bool = False) -> None:
        """
        Click on the predicted location on the minimap.
        """
        # Expand the minimap by hovering over it
        pyautogui.moveTo(self.map_x + self.map_w - 15, self.map_y + self.map_h - 15, duration=0.5)
        sleep(0.5)

        # Click on the predicted location
        pyautogui.click(x, y, duration=0.5)

        if plot:
            self.plot_minimap(x, y)

        # Confirm the location
        pyautogui.click(*self.confirm_button, duration=0.2)
        sleep(2)

        # Move the mouse away to collapse the minimap
        pyautogui.moveTo(self.map_x - 50, self.map_y - 50, duration=0.5)

    def plot_minimap(self, x: int = None, y: int = None) -> None:
        """
        Plot the minimap with reference points and the predicted location.
        """
        minimap = pyautogui.screenshot(region=self.minimap_xywh)
        plot_kodiak_x = self.kodiak_x - self.map_x
        plot_kodiak_y = self.kodiak_y - self.map_y
        plot_hobart_x = self.hobart_x - self.map_x
        plot_hobart_y = self.hobart_y - self.map_y

        plt.imshow(minimap)
        plt.plot(plot_hobart_x, plot_hobart_y, 'ro')
        plt.plot(plot_kodiak_x, plot_kodiak_y, 'ro')
        if x and y:
            plt.plot(x - self.map_x, y - self.map_y, 'bo')

        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/minimap.png")