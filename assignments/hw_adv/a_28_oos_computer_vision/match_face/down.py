from sklearn.datasets import fetch_lfw_people
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# This downloads LFW (200MB+) automatically to your ~/scikit_learn_data folder
# and loads it into numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

print(f"Loaded {lfw_people.images.shape[0]} images")