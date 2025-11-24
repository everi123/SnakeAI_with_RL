from agent import Agent
from snake_env import SnakeGameAI
from utils import load_config
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from IPython.display import clear_output
import os

CONFIG_PATH = './configs/dqn_baseline.yaml'
CONFIG = load_config(CONFIG_PATH)

def train(num_games=None, plot_interval=50):
    if num_games is None:
        num_games = CONFIG.get('NUM_GAMES', 200)
    
    # 1. Get the Run Name from Config
    run_name = CONFIG.get('EXPERIMENT_NAME', 'default') 

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI(headless=True)
    
    print(f"🚀 Starting Training: {run_name}")
    pbar = tqdm(total=num_games, desc=f"Training {run_name}")

    while agent.n_games < num_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # 2. Save with Dynamic Name
                filename = f"dqn_{run_name}.pth"
                agent.model.save(file_name=filename, folder_path="./models")

            total_score += score
            mean_score = total_score / agent.n_games
            plot_scores.append(score)
            plot_mean_scores.append(mean_score)
            
            pbar.update(1)
            pbar.set_postfix({'Score': score, 'Record': record})

            if agent.n_games % plot_interval == 0:
                clear_output(wait=True)
                print(f"Run: {run_name} | Game {agent.n_games} | Record: {record}")

    pbar.close()
    
    # Final Save
    filename = f"dqn_{run_name}.pth"
    agent.model.save(file_name=filename, folder_path="./models")
    
    # 3. Save Graph with Dynamic Name
    plt.figure(figsize=(10, 5))
    plt.title(f'Training History: {run_name}')
    plt.plot(plot_scores, label='Score')
    plt.plot(plot_mean_scores, label='Mean Score')
    plt.legend()
    
    graph_path = f'./logs/graphs/{run_name}_curve.png'
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    plt.savefig(graph_path)
    print(f"✅ {run_name} Complete! Model: models/{filename} | Graph: {graph_path}")

    return agent, plot_scores, plot_mean_scores