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
episode_rewards = []
rolling_average_rewards = []
rolling_window_size = 100  # Size of the window for averaging
action_counts = [0, 0]  # Assuming two actions: acceleration and turning

# Main game loop
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
            model_manager.save()  # Save model upon exit
            running = False
            break

    # Get observation
    observation = env.get_observation()
    model_manager.log_activations(observation)  # Log activations

    # Choose action (epsilon-greedy with potential backtracking)
    if random.random() < epsilon:
        # Random action for exploration
        action = [random.uniform(-1, 1), random.choice([-1, 0, 1])]
    else:
        # Model-based action for exploitation
        with torch.no_grad():
            observation_tensor = torch.tensor(observation, dtype=torch.float32)
            action_tensor = model_manager.model(observation_tensor)

            # Introduce small random perturbations
            action = [
                action_tensor[0].item() + random.uniform(-0.1, 0.1),
                action_tensor[1].item() + random.uniform(-0.1, 0.1)
            ]

    # Count selected actions
    if action[0] > 0:
        action_counts[0] += 1  # Acceleration action
    if action[1] != 0:
        action_counts[1] += 1  # Turning action

    # Update position and check collision
    env.update_position(action[0], action[1])
    collision = env.check_collision()

    # Calculate reward
    if collision:
        reward = -10  # Penalty for collision
        env.reset()  # Reset the environment after collision
    else:
        reward = 0.1 * env.ball_speed  # Reward based on speed

    total_reward += reward  # Accumulate reward

    # Store experience
    model_manager.store_experience(
        observation, action, reward, env.get_observation(), collision)

    # Backtracking logic: If the last action led to collision, slightly adjust
    if collision:
        last_action = model_manager.memory[-1][1] if model_manager.memory else action
        action = [max(0, last_action[0] - 0.1),
                  last_action[1] + random.uniform(-0.1, 0.1)]

    # Store modified experience
    model_manager.store_experience(
        observation, action, reward, env.get_observation(), collision)

    # Perform replay to learn from experiences
    model_manager.replay()

    # Update epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Log the total reward and action frequency every episode
    if collision:  # End of episode
        episode_rewards.append(total_reward)

        # Calculate the rolling average
        if len(episode_rewards) >= rolling_window_size:
            avg_reward = sum(
                episode_rewards[-rolling_window_size:]) / rolling_window_size
            rolling_average_rewards.append(avg_reward)
            print(f"Episode: {len(episode_rewards)}, Total Reward: {
                  total_reward}, Rolling Average: {avg_reward:.2f}")

        # Print action frequencies periodically
        if len(episode_rewards) % 100 == 0:
            print(f"Action Frequencies: Acceleration: {
                  action_counts[0]}, Turning: {action_counts[1]}")

        # Reset total reward for the next episode
        total_reward = 0

        # Evaluation logic every 5000 episodes
        if len(episode_rewards) % 5000 == 0:
            evaluate_model(env, model_manager)

# Evaluation function


def evaluate_model(env, model_manager, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        env.reset()
        episode_reward = 0
        while True:
            observation = env.get_observation()
            with torch.no_grad():
                observation_tensor = torch.tensor(
                    observation, dtype=torch.float32)
                action_tensor = model_manager.model(observation_tensor)
                action = [action_tensor[0].item(), action_tensor[1].item()]

            # Update position and check collision
            env.update_position(action[0], action[1])
            collision = env.check_collision()

            # Accumulate rewards
            episode_reward += 0.1 * env.ball_speed  # or however you define reward

            if collision:
                break

        total_reward += episode_reward

    print(f"Evaluation Average Reward: {total_reward / num_episodes:.2f}")


# Clean up
pygame.quit()
