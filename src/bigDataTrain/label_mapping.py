"""
Mapping des labels ESC-50 et FSD50K vers la taxonomie EcoPulse.

Taxonomie entraînée :
- bird
- human
- motor
- rain_wind
- insect
- animal

Important :
- Pas de classe "other" à l'entraînement.
- Les sons non pertinents retournent None et sont exclus du dataset.
- Les instruments de musique sont exclus.
- FSD50K est mappé par labels exacts, pas par sous-chaînes.
"""

from typing import Optional


ESC50_MAP: dict[str, Optional[str]] = {
    # bird
    "chirping_birds": "bird",
    "crow": "bird",

    # human : présence humaine réelle
    "crying_baby": "human",
    "sneezing": "human",
    "clapping": "human",
    "breathing": "human",
    "coughing": "human",
    "laughing": "human",
    "footsteps": "human",
    "snoring": "human",

    # motor / perturbation anthropique
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

    # Hors taxonomie EcoPulse : exclus
    "sea_waves": None,
    "crackling_fire": None,
    "pouring_water": None,
    "toilet_flush": None,
    "clock_tick": None,
    "glass_breaking": None,
    "washing_machine": None,
    "vacuum_cleaner": None,
    "clock_alarm": None,
    "keyboard_typing": None,
    "mouse_click": None,
    "can_opening": None,
    "door_wood_knock": None,
    "door_wood_creaks": None,
    "drinking_sipping": None,
    "brushing_teeth": None,
    "church_bells": None,
    "fireworks": None,
    "hand_saw": None,
}


FSD50K_LABEL_MAP: dict[str, str] = {
    # bird
    "Bird": "bird",
    "Bird_vocalization_and_bird_call_and_bird_song": "bird",
    "Chirp_and_tweet": "bird",
    "Crow": "bird",
    "Gull_and_seagull": "bird",
    "Fowl": "bird",
    "Chicken_and_rooster": "bird",

    # insect
    "Insect": "insect",
    "Cricket": "insect",
    "Buzz": "insect",

    # human : présence humaine réelle
    "Human_voice": "human",
    "Speech": "human",
    "Conversation": "human",
    "Child_speech_and_kid_speaking": "human",
    "Female_speech_and_woman_speaking": "human",
    "Male_speech_and_man_speaking": "human",
    "Whispering": "human",
    "Shout": "human",
    "Yell": "human",
    "Screaming": "human",
    "Singing": "human",
    "Female_singing": "human",
    "Male_singing": "human",
    "Laughter": "human",
    "Chuckle_and_chortle": "human",
    "Giggle": "human",
    "Crying_and_sobbing": "human",
    "Cough": "human",
    "Sneeze": "human",
    "Breathing": "human",
    "Respiratory_sounds": "human",
    "Sigh": "human",
    "Gasp": "human",
    "Applause": "human",
    "Clapping": "human",
    "Cheering": "human",
    "Crowd": "human",
    "Human_group_actions": "human",
    "Walk_and_footsteps": "human",
    "Run": "human",
    "Hands": "human",
    "Finger_snapping": "human",
    "Chewing_and_mastication": "human",
    "Burping_and_eructation": "human",
    "Fart": "human",

    # motor / perturbation anthropique
    "Vehicle": "motor",
    "Motor_vehicle_(road)": "motor",
    "Traffic_noise_and_roadway_noise": "motor",
    "Car": "motor",
    "Car_passing_by": "motor",
    "Race_car_and_auto_racing": "motor",
    "Truck": "motor",
    "Bus": "motor",
    "Motorcycle": "motor",
    "Vehicle_horn_and_car_horn_and_honking": "motor",
    "Train": "motor",
    "Rail_transport": "motor",
    "Subway_and_metro_and_underground": "motor",
    "Aircraft": "motor",
    "Fixed-wing_aircraft_and_airplane": "motor",
    "Boat_and_Water_vehicle": "motor",
    "Engine": "motor",
    "Engine_starting": "motor",
    "Idling": "motor",
    "Accelerating_and_revving_and_vroom": "motor",
    "Siren": "motor",
    "Power_tool": "motor",
    "Drill": "motor",
    "Sawing": "motor",
    "Mechanical_fan": "motor",
    "Printer": "motor",
    "Mechanisms": "motor",

    # rain/wind
    "Rain": "rain_wind",
    "Raindrop": "rain_wind",
    "Thunder": "rain_wind",
    "Thunderstorm": "rain_wind",
    "Wind": "rain_wind",

    # animal
    "Animal": "animal",
    "Wild_animals": "animal",
    "Domestic_animals_and_pets": "animal",
    "Dog": "animal",
    "Bark": "animal",
    "Cat": "animal",
    "Meow": "animal",
    "Purr": "animal",
    "Growling": "animal",
    "Frog": "animal",
    "Livestock_and_farm_animals_and_working_animals": "animal",
}

# Exclusion explicite : ni human, ni other.
# Ces labels polluent le signal "présence humaine réelle".
FSD50K_EXCLUDED_LABELS = {
    "Music",
    "Musical_instrument",
    "Wind_instrument_and_woodwind_instrument",
    "Brass_instrument",
    "Guitar",
    "Electric_guitar",
    "Acoustic_guitar",
    "Bass_guitar",
    "Plucked_string_instrument",
    "Bowed_string_instrument",
    "Keyboard_(musical)",
    "Piano",
    "Organ",
    "Accordion",
    "Harmonica",
    "Harp",
    "Percussion",
    "Drum",
    "Drum_kit",
    "Bass_drum",
    "Snare_drum",
    "Cymbal",
    "Crash_cymbal",
    "Hi-hat",
    "Tambourine",
    "Tabla",
    "Mallet_percussion",
    "Marimba_and_xylophone",
    "Glockenspiel",
    "Gong",
    "Trumpet",
    "Strum",
}


def map_esc50_label(category: str) -> Optional[str]:
    return ESC50_MAP.get(str(category).strip(), None)


def map_fsd50k_labels(labels: str) -> str | None:
    fsd_labels = [label.strip() for label in str(labels).split(",")]

    detected_targets = set()

    for label in fsd_labels:
        target = FSD50K_LABEL_MAP.get(label)
        if target is not None:
            detected_targets.add(target)

    if len(detected_targets) != 1:
        return None

    return next(iter(detected_targets))
