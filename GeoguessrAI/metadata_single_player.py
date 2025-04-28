import pyautogui
import yaml
import os
import io
import base64
import warnings
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from select_regions import get_coords
from geoguessr_bot import GeoBot

warnings.filterwarnings("ignore", category=FutureWarning)

##############################################################################
# Final 50 countries you care about
##############################################################################
TOP_50 = {
    "argentina","australia","austria","belgium","botswana","brazil","bulgaria","cambodia","canada","chile",
    "colombia","croatia","czechia","denmark","finland","france","germany","greece","hungary","india",
    "indonesia","ireland","israel","italy","japan","kenya","lithuania","malaysia","mexico","netherlands",
    "new zealand","nigeria","norway","peru","philippines","poland","portugal","romania","russia","singapore",
    "south africa","south korea","spain","sweden","switzerland","taiwan","thailand","turkey","united kingdom",
    "united states"
}

# Load environment variables for OpenAI
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

##############################################################################
# 1) LICENSE PLATE & ROAD LINE MAPS
##############################################################################

LICENSE_PLATE_COUNTRY_MAP = {
    "eu white": {
        # European white plates (no yellow):
        # (Note: Some of these can have alternative plates, but we keep it simple)
        "austria","belgium","bulgaria","croatia","czechia","denmark","finland","france","germany","greece",
        "hungary","ireland","italy","lithuania","norway","poland","portugal","romania","russia","spain","sweden",
        "switzerland", "netherlands"
    },
    "eu yellow": {
        # Typically Netherlands & UK
        "netherlands","united kingdom"
    },
    "na standard": {
        # North, Central, South American style white plates
        "united states","canada","mexico","argentina","chile","colombia","peru","brazil"
    },
    "na yellow": {
        # Some states in Colombia or US have yellow plates
        "colombia","united states"
    },
    "asia standard": {
        # White-based plates in Asia
        "japan","south korea","taiwan","thailand","malaysia","philippines","india","israel","cambodia", "singapore"
    },
    "asia yellow": {
        # Indonesia sometimes uses black+yellow or yellow on black
        "indonesia"
    },
    "africa standard": {
        # African countries in the list using typical white-based or local plate
        "south africa","kenya","nigeria","botswana"
    },
    "ambiguous": set(),
    "no plate visible": set(),
}

ROAD_LINE_COUNTRY_MAP = {
    "double yellow": {
        # Common in the Americas, some African countries
        "united states","canada","mexico","colombia","chile","argentina","brazil","peru","south africa","kenya"
    },
    "dashed yellow": {
        "united states","canada","mexico","colombia","argentina","chile","norway","finland","south africa","kenya","brazil", "peru"
    },
    "single yellow": {
        "argentina","peru","kenya","brazil", "japan"
    },
    "dashed white": {
        # Common in left-driving or certain EU roads
        "united kingdom","ireland","australia","new zealand","norway","finland","russia","poland","croatia",
        "germany","france","singapore","japan","south africa","sweden", "netherlands", "philippines", "united states"
    },
    "solid white": {
        "japan","south korea","germany","switzerland","austria","france","netherlands","italy","spain","south africa",
        "ireland", "united kingdom"
    },
    "double white": {
        "france","germany","switzerland","austria","italy","spain","portugal","belgium","netherlands","czechia",
        "poland","croatia","hungary","romania","greece","bulgaria","sweden"
    },
    "no center line": set(),
    "unpaved road": {
        # Some well-known unpaved coverage
        "kenya","botswana","argentina","peru","india","russia","brazil","mexico", "nigeria"
    },
    "ambiguous": set(),
    "no road visible": set(),
}

##############################################################################
# 2) NEW METADATA MAPS
##############################################################################

DRIVING_SIDE_MAP = {
    # Known left-driving countries in the top-50
    "left": {
        "united kingdom","ireland","australia","new zealand","japan","south africa","thailand",
        "singapore","malaysia","kenya","india","indonesia"
    },
    # All other top-50 drive on the right
    "right": {
        c for c in TOP_50
        if c not in {
            "united kingdom","ireland","australia","new zealand","japan","south africa","thailand",
            "singapore","malaysia","kenya","india","indonesia"
        }
    },
    "ambiguous": set(),
}

LANDSCAPE_TYPE_MAP = {
    "mountainous": {
        "argentina", "australia", "austria", "chile", "colombia", "croatia", "greece", "india", "italy", "japan", 
        "mexico", "norway", "peru", "portugal", "romania", "slovenia", "south korea", "switzerland", "turkey",
        "united states"
    },
    "flat": {
        "argentina", "belgium", "botswana", "brazil", "cambodia", "canada", "denmark", "hungary", "lithuania", "turkey",
        "netherlands", "nigeria", "poland", "romania", "russia", "south africa", "united states", "united kingdom", "mexico",
        "singapore", "france", "japan", "australia"
    },
    "hilly": {
        "brazil", "czechia", "france", "germany", "ireland", "italy", "japan", "new zealand", "philippines",
        "portugal", "spain", "switzerland", "united kingdom", "united states", "south africa", "peru", "singapore"
    },
    "desert": {
        "australia", "botswana", "chile", "india", "israel", "kenya", "mexico", "south africa", "united states"
    },
    "forest": {
        "argentina", "australia", "canada", "finland", "france", "germany", "japan", "norway", "poland", "russia", 
        "sweden", "switzerland", "united states"
    },
    "ambiguous": set()
}

VEGETATION_TYPE_MAP = {
    "tropical": {
        # Warm/humid climates
        "argentina", "brazil", "botswana", "cambodia", "colombia", "india", "indonesia", "kenya",
        "malaysia", "mexico", "nigeria", "philippines", "singapore", "south africa", "taiwan",
        "thailand", "united states", "vietnam"
    },
    "temperate": {
        # Seasonal climates
        "argentina", "austria", "australia", "belgium", "bulgaria", "canada", "croatia", "czechia", 
        "denmark", "finland", "france", "germany", "hungary", "ireland", "italy", "japan", "lithuania", 
        "netherlands", "new zealand", "norway", "poland", "portugal", "romania", "south korea",
        "spain", "sweden", "switzerland", "united kingdom", "united states", "turkey", "peru", "russia"
    },
    "mediterranean": {
        "argentina", "australia", "france", "greece", "israel", "italy", "portugal", "spain", "turkey", 
        "united states", "chile", "brazil", "croatia"
    },
    "savanna": {
        # Some African grasslands
        "argentina", "botswana", "brazil", "kenya", "nigeria", "south africa", "india", "australia", "mexico"
    },
    "ambiguous": set(),
}

POWER_POLE_MAP = {
    "wooden": {
        # Commonly used in rural and suburban areas
        "united states", "canada", "ireland", "united kingdom", "sweden", "norway", "new zealand",
        "australia", "japan", "south korea", "lithuania", "finland", "philippines", "cambodia", 
        "south africa", "peru", "france", "romania", "hungary"
    },

    "concrete": {
        # Common in Latin America, Europe, and parts of Asia
        "argentina", "brazil", "chile", "colombia", "mexico", "france", "italy", "spain", "japan", "south korea",
        "portugal", "israel", "philippines", "indonesia", "thailand", "malaysia", "poland", "russia"
    },

    "metal": {
        # More common in Eastern Europe, Russia, parts of Asia and the Middle East
        "russia", "poland", "hungary", "israel", "turkey", "romania", "czechia", "bulgaria", "india", "united states",
        "ireland", "japan"
    },

    "ambiguous": set(),
    "no pole visible": set(),  # New category
}

BOLLARD_TYPE_MAP = {
    "european_style": {
        # Distinct tall white or black bollards with reflectors, common in EU
        "germany", "austria", "switzerland", "italy", "france", "czechia", "poland", "spain",
        "portugal", "romania", "bulgaria", "hungary", "croatia", "greece", "denmark", "finland",
        "sweden", "norway", "lithuania", "netherlands", "belgium"
    },

    "australian_style": {
        # Short red/white or black/yellow posts often with a rectangular shape
        "australia", "new zealand"
    },

    "america_style": {
        # Bollards in the US, Canada, Latin America tend to be rare, or flexible plastic/reflective
        "united states", "canada", "mexico", "brazil", "argentina", "chile", "colombia", "peru"
    },

    "asian_style": {
        # Varies but often narrow metal/plastic posts, sometimes yellow/black, dome-top, or striped
        "japan", "south korea", "taiwan", "thailand", "philippines", "malaysia", "indonesia", "india", "cambodia", "singapore", "israel"
    },

    "ambiguous": set()
}


##############################################################################
# 3) ALL_COUNTRIES is just your TOP_50
##############################################################################
ALL_COUNTRIES = TOP_50

##############################################################################
# Helper Functions
##############################################################################
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

##############################################################################
# 4) GPT Prompt Functions
##############################################################################
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
                            "Look at the license plates in this image. Classify the **license plate type** using one of:\n"
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
                            "Look at the road in this image. Classify the **center road line style** using one of:\n"
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

def ask_driving_side_type(screenshot: Image.Image) -> str:
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
                            "Do vehicles in this image appear to drive on the left side or the right side of the road?\n"
                            "Possible answers:\n- left\n- right\n- ambiguous\n\n"
                            "Return only the single word that best describes the driving side."
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

def ask_landscape_type(screenshot: Image.Image) -> str:
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
                            "Look at the overall landscape in this image. Based on the terrain, elevation, and surroundings, "
                            "classify it into **one** of the following landscape types:\n\n"
                            "- **mountainous**: Tall, steep mountains or dramatic peaks in the background or foreground.\n"
                            "- **hilly**: Rolling or undulating terrain with visible hills, but not as steep or high as mountains.\n"
                            "- **flat**: Very little elevation change; wide open areas, plains, or farmland.\n"
                            "- **desert**: Dry and barren environment, sparse vegetation, dusty or sandy terrain, often yellow or reddish tones.\n"
                            "- **forest**: Densely wooded areas or tree-covered terrain dominating the image.\n"
                            "- **ambiguous**: If the landscape is not clearly visible or doesn’t fit cleanly into any of the above.\n\n"
                            "Focus on topography and vegetation density. If you're uncertain, return 'ambiguous'. "
                            "Return only the single best-matching category."
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


def ask_vegetation_type(screenshot: Image.Image) -> str:
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
                            "Describe the vegetation climate zone of this landscape. Classify into one of:\n"
                            "- tropical\n"
                            "- temperate\n"
                            "- mediterranean\n"
                            "- savanna\n"
                            "- ambiguous\n\n"
                            "Return only the single category."
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

def ask_power_pole_type(screenshot: Image.Image) -> str:
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
                            "Look closely at any visible utility or power poles in this image. "
                            "Determine the material and construction style of the most clearly visible pole. "
                            "Classify it into one of the following categories:\n\n"
                            "- **wooden**: natural grain texture, cylindrical shape, sometimes darker or weathered\n"
                            "- **concrete**: uniform gray or white color, often thicker and more angular, may have grooves or seams\n"
                            "- **metal**: shiny or dull metal surface, thin or lattice-style, sometimes painted or galvanized\n"
                            "- **ambiguous**: poles are too far away, partially visible, or not easily identifiable\n"
                            "- **no pole visible**: no utility or power pole appears in the image\n\n"
                            "Return only one category exactly as written above."
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


def ask_bollard_type(screenshot: Image.Image) -> str:
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
                            "Check for roadside bollards in this image. "
                            "If visible, classify them into one of the following categories:\n"
                            "- european_style (tall, often white or black with reflectors)\n"
                            "- australian_style (short, often red/white or black/yellow posts)\n"
                            "- america_style (small reflective poles or flexible plastic posts, or none at all)\n"
                            "- asian_style (short metal or plastic, sometimes striped or domed)\n"
                            "- ambiguous (if no bollards are visible or it's unclear)\n\n"
                            "Return only the single best-matching category."
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


##############################################################################
# 5) MAIN PLAY TURN LOGIC
##############################################################################
def play_turn(bot: GeoBot, plot: bool = False):
    screenshot = pyautogui.screenshot(region=bot.screen_xywh)

    # 1) License Plate Classification
    try:
        plate_raw = ask_license_plate_type(screenshot)
        plate_info = clean_category(plate_raw, LICENSE_PLATE_COUNTRY_MAP.keys())
    except Exception as e:
        print("Failed license plate check:", e)
        plate_info = "ambiguous"
    print("License Plate:", plate_info)

    # 2) Road Line Classification
    try:
        road_raw = ask_road_line_type(screenshot)
        road_info = clean_category(road_raw, ROAD_LINE_COUNTRY_MAP.keys())
    except Exception as e:
        print("Failed road line check:", e)
        road_info = "ambiguous"
    print("Road Line:", road_info)

    # 3) Driving Side
    try:
        drive_raw = ask_driving_side_type(screenshot)
        drive_info = clean_category(drive_raw, DRIVING_SIDE_MAP.keys())
    except Exception as e:
        print("Failed driving side check:", e)
        drive_info = "ambiguous"
    print("Driving Side:", drive_info)

    # 4) Landscape Type
    try:
        land_raw = ask_landscape_type(screenshot)
        land_info = clean_category(land_raw, LANDSCAPE_TYPE_MAP.keys())
    except Exception as e:
        print("Failed landscape type check:", e)
        land_info = "ambiguous"
    print("Landscape Type:", land_info)

    # 5) Vegetation Type
    try:
        veg_raw = ask_vegetation_type(screenshot)
        veg_info = clean_category(veg_raw, VEGETATION_TYPE_MAP.keys())
    except Exception as e:
        print("Failed vegetation check:", e)
        veg_info = "ambiguous"
    print("Vegetation Type:", veg_info)

    # 6) Power Pole Type
    try:
        pole_raw = ask_power_pole_type(screenshot)
        pole_info = clean_category(pole_raw, POWER_POLE_MAP.keys())
    except Exception as e:
        print("Failed power pole check:", e)
        pole_info = "ambiguous"
    print("Power Pole Type:", pole_info)

    # 7) Bollard Type
    try:
        bollard_raw = ask_bollard_type(screenshot)
        bollard_info = clean_category(bollard_raw, BOLLARD_TYPE_MAP.keys())
    except Exception as e:
        print("Failed bollard check:", e)
        bollard_info = "ambiguous"
    print("Bollard Type:", bollard_info)

    # Construct sets for each metadata
    plate_countries   = LICENSE_PLATE_COUNTRY_MAP.get(plate_info, ALL_COUNTRIES)
    road_countries    = ROAD_LINE_COUNTRY_MAP.get(road_info, ALL_COUNTRIES)
    drive_countries   = DRIVING_SIDE_MAP.get(drive_info, ALL_COUNTRIES)
    land_countries    = LANDSCAPE_TYPE_MAP.get(land_info, ALL_COUNTRIES)
    veg_countries     = VEGETATION_TYPE_MAP.get(veg_info, ALL_COUNTRIES)
    pole_countries    = POWER_POLE_MAP.get(pole_info, ALL_COUNTRIES)
    bollard_countries = BOLLARD_TYPE_MAP.get(bollard_info, ALL_COUNTRIES)

    # Intersect all sets
    metadata_sets = [
        plate_countries, road_countries, drive_countries,
        land_countries, veg_countries, pole_countries, bollard_countries
    ]
    possible_countries = ALL_COUNTRIES
    for mset in metadata_sets:
        if mset:  # only intersect if the set is non-empty
            possible_countries = possible_countries.intersection(mset)

    # Fallback if intersection ends up empty
    if not possible_countries:
        print("All metadata sets conflict — fallback to ALL_COUNTRIES.")
        possible_countries = ALL_COUNTRIES

    print("Final possible countries:", possible_countries)

    # === Custom Rule: Exclude US if no plate and dashed white ===
    if plate_info == "no plate visible" and road_info == "dashed white":
        if "united states" in possible_countries:
            print("Custom Rule Applied: Excluding United States due to no plate + dashed white")
            possible_countries = possible_countries - {"united states"}
    
    # === Custom Rule: Exclude US if no plate and no center line ===
    if plate_info == "no plate visible" and road_info == "no center line":
        if "united states" in possible_countries:
            print("Custom Rule Applied: Excluding United States due to no plate + no center line")
            possible_countries = possible_countries - {"united states"}

    # 8) Model prediction
    resized_screenshot = screenshot.resize(
        (screenshot.width // 4, screenshot.height // 4),
        Image.Resampling.LANCZOS
    )
    predictions = bot.predict_local_ranked(resized_screenshot)

    # PRINT THE TOP 10 GUESSES
    print("\nTop 10 Model Predictions:")
    for rank, (country, prob, coords) in enumerate(predictions[:10], 1):
        print(f"  {rank}. {country:20s} — {prob*100:.2f}% — {coords}")

    # Pick the first predicted country that intersects
    final_guess = None
    for country, prob, coords in predictions:
        if country in possible_countries:
            final_guess = coords
            print(f"Chosen Guess: {country} ({prob:.2f})")
            break

    # If none matched, fallback to top
    if not final_guess and predictions:
        final_guess = predictions[0][2]
        print(f"No intersection match—fallback to top guess: {predictions[0][0]}")

    # 9) Click on map
    if final_guess:
        x, y = bot.lat_lon_to_mercator_map_pixels(*final_guess)
        bot.select_map_location(x, y, plot=plot)
    else:
        print("No final guess found. Clicking center of minimap.")
        default_x = bot.map_x + bot.map_w // 2
        default_y = bot.map_y + bot.map_h // 2
        bot.select_map_location(default_x, default_y, plot=plot)

    pyautogui.press(" ")
    sleep(10)

##############################################################################
# 6) MAIN
##############################################################################
def main(turns=5, plot=False):
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
