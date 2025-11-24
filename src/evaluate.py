import torch
import numpy as np
import os
from agent import Agent
from snake_env import SnakeGameAI
from utils import save_metrics, load_config

# Load config generally, but we might override it
CONFIG = load_config('./configs/dqn_baseline.yaml')

def evaluate(model_path=None, num_games=10):
    """
    Evaluates a trained model.
    Args:
        model_path (str, optional): Specific path to a .pth file. 
                                    If None, loads from the Config's EXPERIMENT_NAME.
        num_games (int): How many games to play.
    """
    
    # 1. Determine filenames (Config vs Manual Override)
    if model_path is None:
        # Case A: Use the Config (Standard workflow)
        run_name = CONFIG.get('EXPERIMENT_NAME', 'default')
        model_path = f"./models/dqn_{run_name}.pth"
    else:
        # Case B: Manual Override (Testing specific files)
        # We try to extract the run name from the filename for logging
        # e.g., "./models/dqn_v0_test.pth" -> "v0_test"
        filename = os.path.basename(model_path)
        run_name = filename.replace("dqn_", "").replace(".pth", "")
    
    metrics_filename = f"./logs/{run_name}_metrics.json"
    
    print(f"🔍 Evaluating Run: {run_name}")
    print(f"📂 Loading Model: {model_path}")
    
    game = SnakeGameAI(headless=True)
    agent = Agent()
    
    if torch.cuda.is_available():
        map_location = None
    else:
        map_location = torch.device('cpu')
        
    try:
        agent.model.load_state_dict(torch.load(model_path, map_location=map_location))
        agent.model.eval()
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}. Please train first.")
        return

    scores = []
    
    # 2. Evaluation Loop
    for i in range(num_games):
        game.reset()
        score = 0
        done = False
        while not done:
            state = agent.get_state(game)
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = agent.model(state0)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1
            reward, done, score = game.play_step(final_move)
        scores.append(score)
        # print(f"Game {i+1}: {score}") # Optional: Un-comment to see every game

    # 3. Calculate Stats
    stats = {
        "experiment_name": run_name,
        "mean_score": float(np.mean(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "std_dev": float(np.std(scores)),
        "raw_scores": scores
    }
    
    # 4. Save Metrics
    save_metrics(metrics_filename, stats)
    print(f"📊 Results saved to {metrics_filename}")
    print(f"   Mean: {stats['mean_score']} | Max: {stats['max_score']}")

if __name__ == "__main__":
    evaluate()
