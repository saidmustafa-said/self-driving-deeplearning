# model.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ModelManager:
    def __init__(self):
        self.model = CarNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95  # Discount factor

    def load(self, path="car_model.pth"):
        """Load model parameters."""
        self.model.load_state_dict(torch.load(path))

    def save(self, path="car_model.pth"):
        """Save model parameters."""
        torch.save(self.model.state_dict(), path)

    def store_experience(self, observation, action, reward, next_observation, done):
        """Store experience in memory."""
        self.memory.append(
            (observation, action, reward, next_observation, done))

    def replay(self):
        """Train the model using experience replay."""
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for observation, action, reward, next_observation, done in batch:
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            next_observation_tensor = torch.tensor(
                next_observation, dtype=torch.float32)

            # Predict Q values
            target = self.model(observation_tensor)
            with torch.no_grad():
                next_target = self.model(next_observation_tensor)
                target_action = reward_tensor + \
                    (self.gamma * torch.max(next_target)) * (1 - int(done))

            # Update the model
            loss = self.criterion(target, action_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
