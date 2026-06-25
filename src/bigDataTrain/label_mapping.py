"""
Mapping des labels ESC-50 et FSD50K vers la taxonomie EcoPulse.

Taxonomie :
- bird
- human
- motor
- rain_wind
- insect
- animal
- other
"""

import re


ESC50_MAP = {
    # bird
    "chirping_birds": "bird",
    "crow": "bird",

    # human
    "crying_baby": "human",
    "sneezing": "human",
    "clapping": "human",
    "breathing": "human",
    "coughing": "human",
    "laughing": "human",
    "footsteps": "human",

    # motor / machine
    "engine": "motor",
    "helicopter": "motor",
    "chainsaw": "motor",
    "siren": "motor",
    "train": "motor",
    "airplane": "motor",
    "car_horn": "motor",

    # rain/wind
    "rain": "rain_wind",
    "wind": "rain_wind",
    "thunderstorm": "rain_wind",

    # insect
    "crickets": "insect",

    # animal
    "dog": "animal",
    "rooster": "animal",
    "pig": "animal",
    "cow": "animal",
    "frog": "animal",
    "cat": "animal",
    "hen": "animal",
    "sheep": "animal",

    # volontairement en other si pas utile terrain
    "sea_waves": "other",
    "crackling_fire": "other",
    "pouring_water": "other",
    "toilet_flush": "other",
    "clock_tick": "other",
    "glass_breaking": "other",
    "washing_machine": "other",
    "vacuum_cleaner": "other",
    "clock_alarm": "other",
    "keyboard_typing": "other",
    "mouse_click": "other",
    "can_opening": "other",
    "door_wood_knock": "other",
    "door_wood_creaks": "other",
    "drinking_sipping": "other",
    "brushing_teeth": "other",
    "snoring": "human",
    "church_bells": "other",
    "fireworks": "other",
    "hand_saw": "other",
}


FSD50K_KEYWORDS = {
    "bird": [
        "bird", "birds", "chirp", "chirping", "tweet", "caw", "crow",
        "sparrow", "pigeon", "owl", "duck", "goose", "seagull",
    ],
    "human": [
        "speech", "conversation", "talking", "voice", "shout", "scream",
        "laugh", "laughter", "cry", "crying", "cough", "sneeze",
        "footstep", "footsteps", "clap", "clapping", "singing",
        "whisper", "breathing", "snoring","Music","Wind_instrument"
    ],
    "motor": [
        "engine", "motor", "vehicle", "car", "truck", "bus", "motorcycle",
        "helicopter", "aircraft", "airplane", "train", "chainsaw",
        "siren", "traffic", "machinery", "machine", "lawn mower",
    ],
    "rain_wind": [
        "rain", "raindrop", "wind", "storm", "thunder", "thunderstorm",
        "weather", "gust",
    ],
    "insect": [
        "insect", "insects", "cricket", "crickets", "bee", "bees",
        "mosquito", "fly", "flies", "buzz", "buzzing", "cicada",
    ],
    "animal": [
        "dog", "bark", "cat", "meow", "cow", "moo", "horse", "neigh",
        "sheep", "pig", "frog", "rooster", "hen", "goat", "animal",
        "livestock",
    ],
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_esc50_label(category: str) -> str:
    category = str(category).strip()
    return ESC50_MAP.get(category, "other")


def map_fsd50k_labels(labels: str) -> str:
    """
    FSD50K contient souvent plusieurs labels par fichier.
    On classe par priorité pour éviter que 'animal' avale 'bird' ou 'insect'.
    """
    text = normalize_text(labels)

    priority = [
        "bird",
        "insect",
        "human",
        "motor",
        "rain_wind",
        "animal",
    ]

    for target in priority:
        for kw in FSD50K_KEYWORDS[target]:
            if normalize_text(kw) in text:
                return target

    return "other"
