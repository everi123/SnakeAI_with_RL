import os
import json
import yaml  # This is the library we just installed
import torch
import numpy as np
import random

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
        else:
            print(f"❌ Error: Unsupported file extension for {path}")
            return None
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def save_metrics(path, data):
    """Saves training metrics to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

class VideoRecorder:
    def __init__(self):
        pass
