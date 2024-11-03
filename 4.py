import pygame
import sys
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

# Initialize Pygame
pygame.init()

# Define display dimensions and colors
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (50, 50, 50)
TRACK_COLOR = (200, 200, 200)
WALL_COLOR = (100, 100, 100)
CAR_COLOR = (255, 0, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Track with Car Movement")

# Track boundaries
track_center = (WIDTH // 2, HEIGHT // 2)
outer_radius = 200
inner_radius = 160

# Car properties
car_width, car_height = 20, 10
car_x, car_y = WIDTH // 2, HEIGHT // 2 - outer_radius + 10  # Start position
car_speed = 0  # Initial speed
car_angle = 0  # Initial angle

# Car movement parameters
max_speed = 5
acceleration = 0.1
deceleration = 0.1
turn_speed = 3

# Neural Network Model


class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize model, optimizer, and loss function
model = CarNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Experience Replay
memory = deque(maxlen=10000)
batch_size = 64
gamma = 0.95  # Discount factor

# Function to reset car


def reset_car():
    global car_x, car_y, car_speed, car_angle
    car_x, car_y = WIDTH // 2, HEIGHT // 2 - outer_radius + 10
    car_speed = 0
    car_angle = 0

# Function to choose action


def get_action(observation, epsilon):
    if random.random() < epsilon:
        # Exploration: choose random action
        return random.uniform(-1, 1), random.uniform(-1, 1)
    else:
        # Exploitation: choose action based on the model
        with torch.no_grad():
            output = model(torch.tensor(observation, dtype=torch.float32))
            return output[0].item(), output[1].item()

# Function to store experience in memory


def store_experience(observation, action, reward, next_observation, done):
    memory.append((observation, action, reward, next_observation, done))

# Function to train the model from experience


def replay():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    for observation, action, reward, next_observation, done in batch:
        observation_tensor = torch.tensor(observation, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        next_observation_tensor = torch.tensor(
            next_observation, dtype=torch.float32)

        # Predict Q values
        target = model(observation_tensor)
        with torch.no_grad():
            next_target = model(next_observation_tensor)
            target_action = reward_tensor + \
                (gamma * torch.max(next_target)) * (1 - int(done))

        # Update the model
        loss = criterion(target, action_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# Game loop
running = True
clock = pygame.time.Clock()
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay rate
epsilon_min = 0.01  # Minimum exploration rate

while running:
    screen.fill(BACKGROUND_COLOR)
    pygame.draw.circle(screen, TRACK_COLOR, track_center, outer_radius, 40)
    pygame.draw.circle(screen, WALL_COLOR, track_center, outer_radius, 5)
    pygame.draw.circle(screen, WALL_COLOR, track_center, inner_radius, 5)

    # Get observation
    observation = [car_x, car_y, car_angle, car_speed]

    # Choose action
    acceleration, turn_angle = get_action(observation, epsilon)
    car_speed = min(max_speed, max(-max_speed, car_speed + acceleration))
    car_angle += turn_angle * turn_speed

    # Update car position
    radian_angle = math.radians(car_angle)
    car_x += car_speed * math.cos(radian_angle)
    car_y -= car_speed * math.sin(radian_angle)

    # Calculate reward
    reward = car_speed * 0.1
    distance_from_center = math.sqrt(
        (car_x - track_center[0]) ** 2 + (car_y - track_center[1]) ** 2)
    done = False

    if distance_from_center > outer_radius or distance_from_center < inner_radius:
        reward = -10
        done = True
        reset_car()

    # Store experience
    next_observation = [car_x, car_y, car_angle, car_speed]
    store_experience(observation, (acceleration, turn_angle),
                     reward, next_observation, done)
    total_reward = reward if done else total_reward + reward

    # Train from experience replay
    replay()

    # Draw car
    car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
    car_surface.fill(CAR_COLOR)
    rotated_car = pygame.transform.rotate(car_surface, -car_angle)
    rotated_rect = rotated_car.get_rect(center=(car_x, car_y))
    screen.blit(rotated_car, rotated_rect.topleft)

    # Display reward
    font = pygame.font.Font(None, 36)
    reward_text = font.render(
        f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
    screen.blit(reward_text, (10, 10))

    # Update display
    pygame.display.flip()
    clock.tick(60)

    # Decay epsilon
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

pygame.quit()
sys.exit()
