# project/model.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class ModelAlpha(nn.Module):
    def __init__(self):
        super(ModelAlpha, self).__init__()
        self.fc1 = nn.Linear(10, 256)
        self.bn1 = nn.LayerNorm(256)

        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.LayerNorm(128)
        self.dropout3 = nn.Dropout(0.2)

        # Output layers: separate for acceleration and turning
        # Output for acceleration (0 or 1)
        self.acceleration_output = nn.Linear(128, 1)
        # Output for turn (-1, 0, 1)
        self.turn_output = nn.Linear(128, 3)

        for layer in [self.fc1, self.fc2, self.fc3, self.acceleration_output, self.turn_output]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)

        # Compute separate outputs
        accel_out = torch.sigmoid(self.acceleration_output(
            x)).squeeze()  # Sigmoid for binary action
        # Softmax for three-way turn action
        turn_out = F.softmax(self.turn_output(x), dim=-1)

        return accel_out, turn_out


class ModelManager:
    def __init__(self):
        self.model = ModelAlpha()
        self.target_model = ModelAlpha()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.99)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.writer = SummaryWriter()
        self.step = 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # Random action
            accel_action = random.choice([0, 1])  # Acceleration: 0 or 1
            turn_action = random.choice([-1, 0, 1])  # Turn: -1, 0, or 1
            return [accel_action, turn_action]
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                accel_out, turn_out = self.model(state_tensor)

                # Choose actions based on model outputs
                accel_action = 1 if accel_out >= 0.5 else 0  # Threshold for binary decision
                turn_action = turn_out.argmax().item() - 1  # Map 0,1,2 to -1,0,1

                return [accel_action, turn_action]

    def store_experience(self, observation, action, reward, next_observation, done):
        self.memory.append(
            (observation, action, reward, next_observation, done))

    def sample_experiences(self):
        return random.sample(self.memory, self.batch_size)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.sample_experiences()
        for observation, action, reward, next_observation, done in batch:
            observation_tensor = torch.tensor(
                observation, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            next_observation_tensor = torch.tensor(
                next_observation, dtype=torch.float32).unsqueeze(0)

            # Get current Q-values for observation
            current_accel_q_value, current_turn_q_values = self.model(
                observation_tensor)

            # Compute target Q-values for each action type
            with torch.no_grad():
                next_accel_q_value, next_turn_q_values = self.model(
                    next_observation_tensor)

                max_next_accel_q_value = next_accel_q_value  # Acceleration output
                max_next_turn_q_value = next_turn_q_values.max()  # Max value of turn outputs

                # Target Q-values based on rewards and future Q-values
                target_accel_q_value = reward_tensor + \
                    (self.gamma * max_next_accel_q_value * (1 - int(done)))
                target_turn_q_value = reward_tensor + \
                    (self.gamma * max_next_turn_q_value * (1 - int(done)))

            # Calculate loss for each output type
            loss_accel = self.criterion(
                current_accel_q_value, target_accel_q_value)
            target_turn_q_values = current_turn_q_values.clone()
            # Adjust index for turn action
            target_turn_q_values[0][action[1] + 1] = target_turn_q_value

            loss_turn = self.criterion(
                current_turn_q_values, target_turn_q_values)
            loss = loss_accel + loss_turn

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.step += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        if self.step % 100 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        self.scheduler.step()
        self.writer.flush()
