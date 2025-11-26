import os
import json
import yaml 
import torch
import numpy as np
import random
import imageio.v3 as iio
import pygame 
from IPython.display import HTML, display
from base64 import b64encode

def load_config(path):
    """Loads configuration from a YAML or JSON file."""
    if not os.path.exists(path):
        print(f"❌ Error: Config file not found at {path}")
        return None
    try:
        if path.endswith('.yaml') or path.endswith('.yml'):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def save_metrics(path, data):
    """Saves training metrics to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

class VideoRecorder:
    def __init__(self, fps=30): 
        self.fps = fps
        self.frames = []

    def capture_frame(self, display_surface):
        view = pygame.surfarray.array3d(display_surface)
        view = view.transpose([1, 0, 2])
        self.frames.append(view)

    def save_video(self, filename="output.mp4", folder_path="./videos"):
        if not self.frames:
            print("❌ No frames to save!")
            return
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, filename)
        try:
            iio.imwrite(path, self.frames, fps=self.fps)
            print(f"✅ Video saved to {path}")
        except Exception as e:
            print(f"❌ Failed to save video: {e}")
        self.frames = []

def display_video(video_path, width=500):
    """Embeds an MP4 video in the notebook."""
    if os.path.exists(video_path):
        mp4 = open(video_path, 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        display(HTML(f"""
        <video width={width} controls autoplay loop>
              <source src="{data_url}" type="video/mp4">
        </video>
        """))
    else:
        print(f"❌ Could not find video at {video_path}")
