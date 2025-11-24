
# SnakeAI with Reinforcement Learning

This project applies **Deep Q-Learning (DQN)** to train an AI agent to play the classic Snake game. The agent learns through trial and error, receiving rewards for eating food and penalties for losing, ultimately aiming to maximize its score. This repository tracks experiments and provides a clean, modular codebase for understanding and extending the DQN approach to Snake.

---

## 📂 Project Structure

A well-organized project is key to managing complexity. Here's the directory structure for SnakeAI:

```
SnakeAI/
├── src/                  # Core Python source code (game logic, agent, training)
│   ├── agent.py          # Implements the DQN agent's logic (model, memory, training steps)
│   ├── model.py          # Defines the neural network architecture for the Q-function
│   ├── snake_env.py      # The Snake game environment (rules, state, rewards)
│   ├── train.py          # Orchestrates the training process for the DQN agent
│   ├── utils.py          # Helper functions (plotting, config loading)
├── models/               # Stores trained PyTorch models (e.g., `dqn_v0_test.pth`)
├── logs/                 # Contains training/evaluation logs and generated graphs
├── configs/              # YAML configuration files for hyperparameters (e.g., `dqn_baseline.yaml`)
└── README.md             # Project overview
```

---

## 💡 Core Components & Flow

The SnakeAI project is built around several interconnected components:

1.  **Snake Game Environment (`src/snake_env.py`):**
    *   Defines the rules of the game: how the snake moves, eats food, and conditions for game over.
    *   Provides the current 'state' of the game to the AI agent.
    *   Calculates 'rewards' or 'penalties' based on the agent's actions and outcomes.

2.  **DQN Agent (`src/agent.py` & `src/model.py`):**
    *   This is the "brain" of the AI.
    *   `model.py` defines the **Q-Network**, a neural network that learns to predict the optimal action (highest Q-value) for any given game state.
    *   `agent.py` manages the agent's behavior:
        *   **Action Selection:** Decides actions based on the Q-Network's predictions, balancing exploration (trying new things) and exploitation (using learned knowledge).
        *   **Replay Memory:** Stores past experiences (state, action, reward, next state, done) to learn efficiently from batches of data.
        *   **Training:** Uses these stored experiences to update the Q-Network's weights.

3.  **Training (`src/train.py`):**
    *   Orchestrates the entire learning process.
    *   Runs many episodes (games) of Snake, allowing the agent to interact with the environment.
    *   Feeds game states, actions, rewards, and next states into the agent's learning mechanism.
    *   Periodically saves trained models and generates performance logs/graphs.

4.  **Evaluation (`src/evaluate.py`):**
    *   After training, this script tests the agent's performance on new games.
    *   Measures the agent's learned skill objectively without exploration.

5.  **Configuration (`configs/dqn_baseline.yaml`):**
    *   Stores all tunable parameters (hyperparameters) for the DQN agent and the training process.
    *   Allows for easy experimentation with different settings without changing the code.

---

## ⚙️ Pipeline

The workflow follows these stages:

1.  **Setup** – Mount Drive in Colab, install dependencies, configure GitHub.
2.  **Config** – Define hyperparameters in YAML configs (`dqn_baseline.yaml` for V0, `dqn_v1.yaml` for V1).
3.  **Train** – Run `train.py` to train the agent. Example (V1):
    ```bash
    python src/train.py --config configs/dqn_v1.yaml
    ```
4.  **Analyze** – Evaluate with `evaluate.py`. Metrics saved in JSON (V1: mean score 35.5, max score 65).
5.  **Save** – Models stored in `models/`, logs and graphs in `logs/`.
6.  **Push** – Commit and push changes to GitHub for version tracking.

---

## 📈 Evolution of Experiments

-   **V0 (Baseline)**: Initial run to validate the environment and agent setup. Produced `dqn_v0_test.pth`. Limited documentation and analysis.
-   **V1 (Improved Run)**: Enhanced experiment naming (`v1_test`), structured configurations, added metrics JSON outputs, generated training graphs, and clearer pipeline documentation.

---

## ▶️ Usage

To run the SnakeAI project:

```bash
# Example: Train agent with a specific configuration
python src/train.py --config configs/dqn_baseline.yaml

# Example: Evaluate a trained model
python src/evaluate.py --model models/dqn_v1_test.pth
```
