# project/main.py
import pygame
import sys
import random
import torch
from track_environment import BallEnvironment
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
    model_manager.log_activations(observation)  # Log activations once

    # Choose action (epsilon-greedy)
    if random.random() < epsilon:
        acceleration = random.uniform(-1, 1)
        turn_angle = random.uniform(-1, 1)
    else:
        with torch.no_grad():
            observation_tensor = torch.tensor(
                observation, dtype=torch.float32)
            action = model_manager.model(observation_tensor)
            acceleration, turn_angle = action[0].item(), action[1].item()

    # Clamp action values to prevent extreme actions
    acceleration = max(-1, min(acceleration, 1))
    turn_angle = max(-1, min(turn_angle, 1))

    # Update environment
    env.update_position(acceleration, turn_angle)
    reward = 0.1 * min(env.ball_speed, env.ball_speed)  # Cap reward
    done = env.check_collision()
    total_reward += reward

    # Store experience
    next_observation = env.get_observation()
    model_manager.store_experience(
        observation, (acceleration, turn_angle), reward, next_observation, done)

    # Replay experience for training
    model_manager.replay()

    # Render environment
    env.render(total_reward)

    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Cap FPS
    clock.tick(60)

pygame.quit()
sys.exit()
