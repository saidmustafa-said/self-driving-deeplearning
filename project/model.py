from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # 3 inputs for the observation
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # Output: acceleration and turn angle

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
        self.gamma = 0.95
        self.writer = SummaryWriter()  # Add TensorBoard writer
        self.step = 0  # Initialize step counter
        self.log_activations_once = False

    def log_activations(self, observation):
        if not self.log_activations_once:
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            self.writer.add_graph(self.model, observation_tensor)
            self.log_activations_once = True

    def load(self, path="car_model.pth"):
        """Load model parameters."""
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def save(self, path="car_model.pth"):
        """Save model parameters."""
        torch.save(self.model.state_dict(), path)

    def store_experience(self, observation, action, reward, next_observation, done):
        """Store experience in memory."""
        self.memory.append(
            (observation, action, reward, next_observation, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch_loss = 0
        for observation, action, reward, next_observation, done in batch:
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            next_observation_tensor = torch.tensor(
                next_observation, dtype=torch.float32)

            # Get current Q-values and next Q-values
            current_q_values = self.model(observation_tensor)
            with torch.no_grad():
                next_q_values = self.model(next_observation_tensor)
                max_next_q_value = torch.max(next_q_values)
                target_q_value = reward_tensor + \
                    (self.gamma * max_next_q_value * (1 - int(done)))

            # Update only the Q-value corresponding to the chosen action
            target_q_values = current_q_values.clone()
            target_q_values[0] = target_q_value if action[0] > 0 else current_q_values[0]
            target_q_values[1] = target_q_value if action[1] != 0 else current_q_values[1]

            # Calculate loss and perform backpropagation
            loss = self.criterion(current_q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()

            # Sum loss for logging
            batch_loss += loss.item()

        # Log the average loss for the batch to TensorBoard
        self.writer.add_scalar("Loss/train", batch_loss /
                               self.batch_size, global_step=self.step)
        self.step += 1
        self.writer.flush()
