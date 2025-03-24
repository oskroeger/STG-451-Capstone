import pyautogui
import yaml
import os
import io
import base64
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from select_regions import get_coords
from geoguessr_bot import GeoBot

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

LICENSE_PLATE_COUNTRY_MAP = {
    "eu white": {
        # All EU/Schengen/nearby countries that do NOT use yellow plates
        "albania", "andorra", "austria", "belgium", "bosnia and herzegovina", "bulgaria", "croatia",
        "czechia", "denmark", "estonia", "finland", "france", "germany", "greece", "hungary", "iceland",
        "ireland", "italy", "kosovo", "latvia", "liechtenstein", "lithuania", "malta", "monaco", "montenegro",
        "north macedonia", "norway", "poland", "portugal", "romania", "san marino", "serbia", "slovakia",
        "slovenia", "spain", "sweden", "switzerland", "vatican city"
    },
    "eu yellow": {"netherlands", "luxembourg", "united kingdom"},
    "na standard": {
        # Most of North and South America with standard-style white plates
        "united states", "canada", "mexico", "brazil", "argentina", "chile", "peru", "paraguay", "uruguay",
        "bolivia", "venezuela", "ecuador"
    },
    "na yellow": {"colombia", "united states"},
    "asia standard": {
        "japan", "south korea", "taiwan", "thailand", "malaysia", "vietnam", "philippines"
    },
    "asia yellow": {"hong kong", "indonesia"},
    "africa standard": {
        "south africa", "kenya", "ghana", "uganda", "botswana", "morocco", "tunisia"
    },
    "ambiguous": set(),
    "no plate visible": set(),
}

ROAD_LINE_COUNTRY_MAP = {
    "double yellow": {
        "united states", "canada", "mexico", "colombia", "chile", "argentina", "brazil", "ecuador", "peru",
        "south africa", "kenya", "ghana", "morocco", "tunisia"
    },
    "dashed yellow": {
        "united states", "canada", "mexico", "colombia", "brazil", "argentina", "chile",
        "norway", "finland",
        "south africa", "kenya", "morocco"
    },
    "single yellow": {
        "brazil", "argentina", "peru", "bolivia", "vietnam",
        "kenya", "ghana"
    },
    "dashed white": {
        "united kingdom", "ireland", "australia", "new zealand", "norway", "finland", "russia",
        "germany", "france", "singapore", "japan",
        "south africa", "morocco", "tunisia"
    },
    "solid white": {
        "japan", "south korea", "germany", "switzerland", "austria", "france", "netherlands", "italy", "spain",
        "south africa", "morocco"
    },
    "double white": {
        "france", "germany", "switzerland", "austria", "italy", "spain", "portugal", "belgium",
        "netherlands", "czechia", "slovenia", "slovakia", "poland", "croatia", "hungary", "romania",
        "greece", "serbia", "bulgaria", "albania", "bosnia and herzegovina", "montenegro",
        "south africa"
    },
    "no center line": set(),  # Not useful for elimination
    "unpaved road": {
        "mongolia", "kenya", "botswana", "namibia", "madagascar", "brazil", "argentina", "peru", "india", "russia"
    },
    "ambiguous": set(),
    "no road visible": set(),
}

ALL_COUNTRIES = set().union(*LICENSE_PLATE_COUNTRY_MAP.values(), *ROAD_LINE_COUNTRY_MAP.values())

def image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def clean_category(response_text: str, valid_categories: set) -> str:
    """
    Attempts to match GPT response to one of the known categories.
    Falls back to 'ambiguous' if no good match.
    """
    text = response_text.strip().lower()
    for category in valid_categories:
        if category in text:
            return category
    return "ambiguous"


def ask_license_plate_type(screenshot: Image.Image) -> str:
    base64_image = image_to_base64(screenshot)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "If there is a license plate visible, classify it into one of the following categories:\n"
                            "- eu white\n"
                            "- eu yellow\n"
                            "- na standard\n"
                            "- na yellow\n"
                            "- asia standard\n"
                            "- asia yellow\n"
                            "- africa standard\n"
                            "- ambiguous\n"
                            "- no plate visible\n\n"
                            "Return only the category that best describes the license plate in this image."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=20,
    )

    return response.choices[0].message.content.strip().lower()


def ask_road_line_type(screenshot: Image.Image) -> str:
    base64_image = image_to_base64(screenshot)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at the road in this image. Classify the **center road line style** using one of the following categories:\n"
                            "- double yellow\n"
                            "- dashed yellow\n"
                            "- single yellow\n"
                            "- dashed white\n"
                            "- solid white\n"
                            "- no center line\n"
                            "- unpaved road\n"
                            "- ambiguous\n"
                            "- no road visible\n\n"
                            "Return only the category that best describes the road lines in this image."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ]
            }
        ],
        max_tokens=20,
    )

    return response.choices[0].message.content.strip().lower()


def play_turn(bot: GeoBot, plot: bool = False):
    """
    Play a single turn in GeoGuessr using the local model and filtered clues.
    """
    # Capture a screenshot of the game
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)

    # Get license plate category
    try:
        license_plate_raw = ask_license_plate_type(screenshot)
        license_plate_info = clean_category(license_plate_raw, LICENSE_PLATE_COUNTRY_MAP.keys())
        print("License Plate Info:", license_plate_info)
    except Exception as e:
        print("Failed to ask about license plate:", e)
        license_plate_info = "ambiguous"

    # Get road line category
    try:
        road_line_raw = ask_road_line_type(screenshot)
        road_line_info = clean_category(road_line_raw, ROAD_LINE_COUNTRY_MAP.keys())
        print("Road Line Info:", road_line_info)
    except Exception as e:
        print("Failed to ask about road lines:", e)
        road_line_info = "ambiguous"

    print("Raw license plate GPT response:", license_plate_raw)
    print("Raw road line GPT response:", road_line_raw)

    # Filter possible countries based on available clues
    possible_countries = ALL_COUNTRIES.copy()

    plate_countries = ALL_COUNTRIES
    if license_plate_info not in {"no plate visible", "ambiguous"}:
        plate_countries = LICENSE_PLATE_COUNTRY_MAP.get(license_plate_info, ALL_COUNTRIES)

    roadline_countries = ALL_COUNTRIES
    if road_line_info not in {"no road visible", "ambiguous"}:
        raw = ROAD_LINE_COUNTRY_MAP.get(road_line_info, ALL_COUNTRIES)
        if raw:  # Only intersect if it's not an empty set
            roadline_countries = raw

    # Intersect only non-empty sets
    possible_countries = plate_countries & roadline_countries

    # Fallback to union if empty after intersection
    if not possible_countries:
        print("⚠️ No countries matched both clues — using union instead of intersection.")
        possible_countries = plate_countries | roadline_countries

    print("Remaining possible countries:", possible_countries)

    # Resize screenshot for model prediction
    resized_screenshot = screenshot.resize((screenshot.width // 4, screenshot.height // 4), Image.Resampling.LANCZOS)

    # Get model predictions (assumes your bot returns a ranked list of (country, coordinates))
    predictions = bot.predict_local_ranked(resized_screenshot)  # e.g., returns [(country, (lat, lon)), ...]

    print("\nModel Predictions (Top-K):")
    for country, prob, coords in predictions:
        print(f"  {country} ({prob:.2f}) -> {coords}")

    # Filter predictions to match only possible countries
    final_guess = None
    for country, prob, coords in predictions:
        if country in possible_countries:
            final_guess = coords
            print(f"Selected prediction from valid country: {country} ({prob:.2f})")
            break

    # If no valid filtered prediction, fallback to top guess
    if final_guess is None and predictions:
        final_guess = predictions[0][2]  # index 2 = coords
        print("Fallback to top prediction (no match with filtered countries)")

    if final_guess:
        x, y = bot.lat_lon_to_mercator_map_pixels(*final_guess)
        bot.select_map_location(x, y, plot=plot)
    else:
        print("Failed to get coordinates. Clicking center of minimap.")
        default_x = bot.map_x + bot.map_w // 2
        default_y = bot.map_y + bot.map_h // 2
        bot.select_map_location(default_x, default_y, plot=plot)

    # Submit guess and proceed
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