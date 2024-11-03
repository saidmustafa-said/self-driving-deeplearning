import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os


class CarNet(nn.Module):
    def __init__(self, input_size, num_balls):
        super(CarNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Input layer to hidden layer
        # Hidden layer to output layer (2 outputs per ball)
        self.fc2 = nn.Linear(128, num_balls * 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelManager:
    def __init__(self, input_size, num_balls, batch_size=64, gamma=0.99):
        self.model = CarNet(input_size, num_balls)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)  # Replay memory
        self.batch_size = batch_size
        self.gamma = gamma

    def store_experience(self, observation, actions, reward, next_observation, done):
        """Store experience in memory."""
        self.memory.append(
            (observation, actions, reward, next_observation, done))

    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for observation, actions, reward, next_observation, done in batch:
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            actions_tensor = torch.tensor(actions, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            next_observation_tensor = torch.tensor(
                next_observation, dtype=torch.float32)

            # Q-learning update rule
            current_q_values = self.model(observation_tensor)
            with torch.no_grad():
                next_q_values = self.model(next_observation_tensor)
                max_next_q_value = torch.max(next_q_values)
                target_q_values = reward_tensor + \
                    (self.gamma * max_next_q_value * (1 - int(done)))

            # Compute loss and backpropagate
            loss = self.criterion(current_q_values, actions_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def load(self, path):
        """Load model state from a checkpoint."""
        if os.path.isfile(path):
            self.model.load_state_dict(torch.load(path))
            print("Model loaded successfully.")
        else:
            print("No model found, starting fresh.")
