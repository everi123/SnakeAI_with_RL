import os
import sys
from pathlib import Path

def in_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_project(base_dir="MLProject"):
    # If running in Colab, mount Google Drive
    if in_colab():
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = Path(f"/content/drive/MyDrive/{base_dir}")
    else:
        base_path = Path(base_dir)

    # Define folders
    folders = [
        "notebooks",
        "src",
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "configs"
    ]

    # Create folders
    for folder in folders:
        (base_path / folder).mkdir(parents=True, exist_ok=True)

    # Create starter files
    files = {
        "README.md": "# ML Project Starter\n\nThis is a customizable starter template for ML projects.\n",
        "requirements.txt": "numpy\npandas\nscikit-learn\ntorch\nmatplotlib\n",
        "src/__init__.py": "",
        "src/model.py": "class Model:\n    def __init__(self):\n        pass\n    def train(self, data):\n        pass\n    def predict(self, x):\n        pass\n",
        "src/train.py": "def train():\n    # TODO: implement training loop\n    pass\n\nif __name__ == '__main__':\n    train()\n",
        "src/evaluate.py": "def evaluate():\n    # TODO: implement evaluation\n    pass\n",
        "src/api.py": "from flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.route('/predict', methods=['POST'])\ndef predict():\n    data = request.json\n    return jsonify({'prediction': None})\n\nif __name__ == '__main__':\n    app.run(debug=True)\n"
    }

    for filename, content in files.items():
        file_path = base_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

    print(f"✅ Project structure created at {base_path}")

if __name__ == "__main__":
    setup_project("testProj")  # Change name for each new project
