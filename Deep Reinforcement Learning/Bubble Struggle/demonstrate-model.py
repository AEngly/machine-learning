import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import cv2

# Choose episodes
n_episodes = 600

# Define the DQN model (same as in the training script)
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Load the environment
env = gym.make("CartPole-v1", render_mode="human")

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get number of actions and observations
n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

# Initialize the policy network and load the trained model
policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("Models/dqn_cartpole_model_{}.pth".format(n_episodes)))
policy_net.eval()  # Set the network to evaluation mode

# Function to select action based on the current state
def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)

# Function to play the game with visualization
def play_game(num_episodes):
    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for t in count():
            env.render()
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            print(observation, reward, terminated, truncated)
            total_reward += reward
            done = terminated or truncated
            if done:
                print("Episode {} finished after {} timesteps with total reward: {}".format(i_episode+1, t+1, total_reward))
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

# Play 5 episodes
play_game(1)