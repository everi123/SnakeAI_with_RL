import torch
import pygame
from agent import Agent
from snake_env import SnakeGameAI
from utils import VideoRecorder, load_config

CONFIG = load_config('./configs/dqn_baseline.yaml')

def record_demo(model_name="dqn_v1_production.pth"):
    print(f"🎬 Starting Video Recording for {model_name}...")
    
    # 1. Setup
    game = SnakeGameAI(headless=False) # Headless MUST be False to record pixels
    agent = Agent()
    recorder = VideoRecorder(fps=15) # Snake doesn't move that fast
    
    # 2. Load Model
    model_path = f"./models/{model_name}"
    agent.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    agent.model.eval()

    # 3. Play ONE Game
    game.reset()
    done = False
    while not done:
        # Get move
        state = agent.get_state(game)
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = agent.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        
        # Move
        reward, done, score = game.play_step(final_move)
        
        # Capture Frame
        recorder.capture_frame(game.display)
        
        # Optional: Slow down slightly so we can see it live too
        pygame.time.wait(20)

    print(f"🏁 Game Over. Score: {score}")
    recorder.save_video(filename=f"demo_{model_name}.mp4")

if __name__ == "__main__":
    # Make sure you put the name of your BEST model here
    record_demo("dqn_v1_production.pth")