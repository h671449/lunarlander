import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
GAMMA = 0.99 # 0.99
LR = 1e-4 # 1e-4
BATCH_SIZE = 64
MEM_SIZE = 100_000
EPS_START = 1.0
EPS_END = 0.01 # 0.01
EPS_DECAY = 0.995 # 0.995
TARGET_UPDATE = 10 # 10
NUM_EPISODES = 1000
EVAL_EVERY = 50

# Environment
env = gym.make("LunarLander-v3")
state_dim = env.observation_space.shape[0]  # 8
action_dim = env.action_space.n            # 4

# Q-network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert list of numpy arrays to numpy array before converting to tensor
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(device)
        
        return states, actions, rewards, next_states, dones


    def __len__(self):
        return len(self.buffer)

# Initialize
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayBuffer(MEM_SIZE)
epsilon = EPS_START

episode_rewards = []
eval_scores = []

# Training function
def train_step():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # Current Q values
    q_values = policy_net(states).gather(1, actions)

    # Double DQN
    next_actions = policy_net(next_states).argmax(1, keepdim=True)
    next_q_values = target_net(next_states).gather(1, next_actions)

    target_q = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.SmoothL1Loss()(q_values, target_q) # nn.MSELoss()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Evaluation function
def evaluate_agent(episodes=5):
    total = 0
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        total += ep_reward
    return total / episodes

# Training loop
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.push((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        train_step()

    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(EPS_END, epsilon * EPS_DECAY)

    # Update target net
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Evaluate
    if episode % EVAL_EVERY == 0:
        eval_score = evaluate_agent()
        eval_scores.append((episode, eval_score))
        print(f"Episode {episode}, reward: {total_reward:.2f}, eval avg: {eval_score:.2f}, epsilon: {epsilon:.3f}")

# Plot
plt.plot(episode_rewards, label='Training reward')
if eval_scores:
    episodes, scores = zip(*eval_scores)
    plt.plot(episodes, scores, label='Eval avg reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("LunarLander DQN Performance")
plt.legend()
plt.grid()
plt.show()

env.close()
