import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

base_dir = project_root / "data" / "test"

for person in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person)
    if os.path.isdir(person_dir):
        images = os.listdir(person_dir)
        print(person, ":", len(images), "images")
