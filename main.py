# main.py

import pygame
import sys
import random
import torch
from track_environment import BallEnvironment  # Updated import
from model import ModelManager


# Initialize environment and model
env = BallEnvironment()
model_manager = ModelManager()

# Load model if available
try:
    model_manager.load()
    print("Model loaded successfully.")
except FileNotFoundError:
    print("No model found, starting fresh.")

# Game loop parameters
running = True
clock = pygame.time.Clock()
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
total_reward = 0

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            model_manager.save()  # Save model upon exit
            running = False
            break

    # Get observation
    observation = env.get_observation()

    # Choose action (epsilon-greedy)
    if random.random() < epsilon:
        acceleration = random.uniform(-1, 1)
        turn_angle = random.uniform(-1, 1)
    else:
        with torch.no_grad():
            output = model_manager.model(
                torch.tensor(observation, dtype=torch.float32))
            acceleration, turn_angle = output[0].item(), output[1].item()

    # Update car position and check collision
    env.update_position(acceleration, turn_angle)
    done = env.check_collision()
    # Adjust reward based on ball speed
    reward = -10 if done else 0.1 * env.ball_speed
    if done:
        env.reset()

    # Store experience and replay
    next_observation = env.get_observation()
    model_manager.store_experience(
        observation, (acceleration, turn_angle), reward, next_observation, done)
    model_manager.replay()

    # Render environment
    total_reward += reward
    env.render(total_reward)

    # Epsilon decay
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()
