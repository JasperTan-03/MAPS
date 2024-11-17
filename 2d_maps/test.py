import json

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl_environment import SegmentationEnv

# Get image

PATH = "./data/gtFine/train/aachen/aachen_000000_000019_gtFine_color.png"
image = Image.open(PATH)

# Load the polygon label data
LABEL_PATH = "./data/gtFine/train/aachen/aachen_000000_000019_gtFine_polygons.json"
with open(LABEL_PATH) as f:
    polygons = json.load(f)

# Create a mask image
mask = Image.new("L", (polygons["imgWidth"], polygons["imgHeight"]), 0)
draw = ImageDraw.Draw(mask)

# Draw the polygons for the current label
for obj in polygons["objects"]:
    if obj["label"] == "car":

        polygon = obj["polygon"]
        # Convert list of lists to list of tuples
        polygon_tuples = [tuple(point) for point in polygon]
        draw.polygon(polygon_tuples, outline=1, fill=1)

mask_array = np.array(mask)

# Convert PNG image to numpy array
image_array = np.array(image)

# Grey scale the image
grey_image = image.convert("L")
grey_image_array = np.array(grey_image)

mask_array = np.array(mask)

# Create environment
print("Creating Environment...")
env = SegmentationEnv(image=grey_image_array, labels=mask_array, step_limit=1000)

# Run Episode
print("Running Episode...")
obs = env.reset()
total_reward = 0

while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward

    env.render()

    if done:
        print(f"Episode finished!")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Coverage: {info['coverage']*100:.1f}%")
        print(f"Accuracy: {info['accuracy']*100:.1f}%")
        break

plt.ioff()
plt.show()
