import os

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))

base_dir = os.path.join(project_root, "face_dataset")

for person in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person)
    if os.path.isdir(person_dir):
        images = os.listdir(person_dir)
        print(person, ":", len(images), "images")
