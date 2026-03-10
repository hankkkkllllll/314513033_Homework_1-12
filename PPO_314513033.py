import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
from collections import namedtuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

# ==========================================
# Hyperparameters & Configurations
# ==========================================
EVALUATION_PHASE = True   # Set to True to visualize the agent after training
ENV_NAME = 'MountainCar-v0'
GAMMA = 0.99              # Discount factor for future rewards
RENDER_TRAIN = False      # Render environment during training
SEED = 1
LOG_INTERVAL = 10

# Initialize environment and extract dimensions
env = gym.make(ENV_NAME)
num_state = env.observation_space.shape[0]
num_action = env.action_space.n

# Set random seeds for reproducibility
torch.manual_seed(SEED)
env.action_space.seed(SEED)

# Define a namedtuple to store agent experiences
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

# ==========================================
# Neural Network Architectures
# ==========================================
class Actor(nn.Module):
    """
    Actor Network: Determines the policy (action distribution) based on the current state.
    """
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.action_head = nn.Linear(128, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # Output a probability distribution over discrete actions
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

class Critic(nn.Module):
    """
    Critic Network: Evaluates the current state by estimating the expected return (State Value).
    """
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

# ==========================================
# PPO Agent Implementation
# ==========================================
class PPO():
    def __init__(self):
        super(PPO, self).__init__()
        # PPO specific parameters
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 32

        # Initialize networks and optimizers
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-3)

        # Experience buffer and logging
        self.buffer = []
        self.training_step = 0
        self.writer = SummaryWriter('../exp')

        # Create directories for saving models
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        """Samples an action from the Actor's probability distribution."""
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        
        dist = Categorical(action_prob)
        action = dist.sample()
        return action.item(), action_prob[:, action.item()].item()

    def store_transition(self, transition):
        """Saves a transition step into the memory buffer."""
        self.buffer.append(transition)

    def update(self, current_epoch):
        """Updates the Actor and Critic networks using the PPO algorithm."""
        # Convert buffer lists to tensors
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        # Calculate discounted rewards (Returns)
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + GAMMA * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)

        # Optimize policy for K epochs
        for _ in range(self.ppo_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                
                # Print training progress periodically
                if self.training_step % 1000 == 0:
                    print(f'Epoch: {current_epoch} | Training Step: {self.training_step}')

                # Get the true return and value estimation
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                
                # Calculate Advantage
                delta = Gt_index - V
                advantage = delta.detach()

                # Get new action probabilities
                action_prob = self.actor_net(state[index]).gather(1, action[index])
                
                # --- PPO Core Logic --- #
                # 1. Calculate policy ratio
                pi_ratio = action_prob / old_action_log_prob[index]

                # 2. Calculate clipped surrogate objective
                unclipped_adv = pi_ratio * advantage
                clipped_adv = torch.clamp(pi_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                
                # 3. Add entropy bonus to encourage exploration
                action_dist = Categorical(self.actor_net(state[index]))
                policy_entropy = action_dist.entropy().mean()

                # Combine into final Actor Loss
                actor_loss = -(torch.min(unclipped_adv, clipped_adv).mean() + 0.01 * policy_entropy)

                # Actor Backpropagation
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Critic Backpropagation (MSE Loss)
                critic_loss = F.mse_loss(V, Gt_index)
                self.critic_net_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                # ---------------------- #

                self.training_step += 1

        # Clear memory buffer after update
        del self.buffer[:]

# ==========================================
# Main Training Loop
# ==========================================
def main():
    agent = PPO()
    
    for i_epoch in range(1000):
        state, _ = env.reset(seed=SEED)
        if RENDER_TRAIN: env.render()

        for t in count():
            # Agent interacts with the environment
            action, action_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # === Custom Reward Shaping === #
            # Extract physical parameters
            current_pos, current_vel = next_state
            
            # Augment base reward (-1) with kinetic energy (speed) to encourage oscillation
            kinetic_bonus = 100.0 * abs(current_vel)
            custom_reward = reward + kinetic_bonus 
            
            # Proximity bonus: Reward the agent when it reaches the right-side slope
            if current_pos > 0.1:
                custom_reward += 10.0
            
            # Store experience
            trans = Transition(state, action, action_prob, custom_reward, next_state)
            # ============================= #
            
            if RENDER_TRAIN: env.render()
            agent.store_transition(trans)
            state = next_state

            # If episode ends, update the network
            if done:
                if len(agent.buffer) >= agent.batch_size: 
                    agent.update(i_epoch)
                agent.writer.add_scalar('Performance/Steps_per_episode', t, global_step=i_epoch)
                break

    # ==========================================
    # Evaluation Phase
    # ==========================================
    if EVALUATION_PHASE:
        print("\n" + "="*50)
        print("🚀 [Info] Training Completed. Initiating Evaluation...")
        print("="*50 + "\n")
        
        eval_env = gym.make(ENV_NAME, render_mode='human')
        state, _ = eval_env.reset(seed=SEED)
        
        for step in range(500):
            eval_env.render()
            action, _ = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            state = next_state
            
            if terminated or truncated:
                if terminated:
                    print(f"✅ Success! Target reached in {step + 1} steps.")
                else:
                    print(f"⚠️ Truncated! Max evaluation steps reached.")
                break
                
        eval_env.close()

if __name__ == '__main__':
    main()
    print("Script execution finished.")