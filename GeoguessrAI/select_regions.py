import pyautogui
from pynput import keyboard
import yaml

# Define the screen and map regions
regions = [
    "screen_top_left",
    "screen_bot_right",
]

map_regions = [
    "map_top_left",
    "map_bottom_right",
    "confirm_button",
    "kodiak",
    "hobart",
]

next_round_button = "next_round_button"
coords = []

# Key to press for capturing coordinates
PRESS_KEY = 'a'

# Function to handle key press
def on_press(key):
    try:
        if key.char == PRESS_KEY:
            x, y = pyautogui.position()
            print(f"Captured coordinates: ({x}, {y})")
            coords.append([x, y])
            return False  # Stop the listener after capturing
    except AttributeError:
        pass

# Function to get coordinates from user
def get_coords(players=1):
    # Capture screen regions (top left and bottom right)
    for region in regions:
        print(f"Move the mouse to the {region} region and press '{PRESS_KEY}'.")
        with keyboard.Listener(on_press=on_press) as keyboard_listener:
            keyboard_listener.join(timeout=40)

    # Capture map regions for each player
    for p in range(1, players + 1):
        print(f"\nCapturing map regions for player {p}:")
        for region in map_regions:
            specific_region = f"{region}_{p}"
            print(f"Move the mouse to the {specific_region} and press '{PRESS_KEY}'.")
            with keyboard.Listener(on_press=on_press) as keyboard_listener:
                keyboard_listener.join(timeout=40)
            regions.append(specific_region)

    # Capture the "Next Round" button
    print(f"Move the mouse to the {next_round_button} and press '{PRESS_KEY}'.")
    with keyboard.Listener(on_press=on_press) as keyboard_listener:
        keyboard_listener.join(timeout=40)
    regions.append(next_round_button)

    # Create a dictionary of region names and their coordinates
    screen_regions = {reg: coord for reg, coord in zip(regions, coords)}

    # Save the dictionary as a YAML file
    try:
        with open("screen_regions.yaml", "w") as f:
            yaml.dump(screen_regions, f)
        print("\nCoordinates saved successfully to 'screen_regions.yaml'.")
    except Exception as e:
        print(f"Error saving coordinates: {e}")

    return screen_regions

if __name__ == "__main__":
    # Run the script with one player by default
    _ = get_coords(players=1)
