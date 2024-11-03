import pygame
import random
import numpy as np
import torch
from model import ModelManager
from track_environment import BallEnvironment

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
NUM_BALLS = 5


def get_observation(balls):
    """Get the observation vector from the current state of the balls."""
    return np.array([[ball.x, ball.y, ball.speed] for ball in balls]).flatten()


def main():
    clock = pygame.time.Clock()
    model_manager = ModelManager(input_size=3 * NUM_BALLS, num_balls=NUM_BALLS)


    # Initialize the environment
    env = BallEnvironment(num_balls=NUM_BALLS)

    epsilon = 1.0  # Initial epsilon for exploration
    epsilon_decay = 0.995
    epsilon_min = 0.1
    episodes = 1000

    for episode in range(episodes):
        observation = get_observation(env.balls)
        done = False
        total_reward = 0

        while not done:

            # Choose actions for each ball (epsilon-greedy)
            # Choose actions for each ball (epsilon-greedy)
            accel_inputs = []
            turn_inputs = []
            for i in range(NUM_BALLS):
                if random.random() < epsilon:
                    # Random actions for exploration
                    accel_inputs.append(random.uniform(-1, 1))  # Acceleration
                    turn_inputs.append(random.uniform(-2, 2))   # Turning
                else:
                    with torch.no_grad():
                        output = model_manager.model(
                            torch.tensor(observation, dtype=torch.float32))
                        print("Model output shape:", output.shape)  # Debugging line
                        # Get actions for each ball
                        # Acceleration for ball i
                        accel_inputs.append(output[2 * i].item())
                        turn_inputs.append(output[2 * i + 1].item())  # Turn for ball i


            # Apply actions to balls
            for i, ball in enumerate(env.balls):
                ball.acceleration = accel_inputs[i]
                ball.turn = turn_inputs[i]
                ball.move()

            # Render the environment
            env.render()

            # Simulate a reward and check for done condition
            reward = 1  # Reward for staying within bounds, for example
            done = env.check_collisions()
            total_reward += reward

            # Store experience
            next_observation = get_observation(env.balls)
            actions = []
            for i in range(NUM_BALLS):
                actions.extend([accel_inputs[i], turn_inputs[i]])
            model_manager.store_experience(
                observation, actions, reward, next_observation, done)

            # Replay the experiences
            model_manager.replay()

            observation = next_observation

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {
              episode + 1}/{episodes} completed with total reward: {total_reward}")

    pygame.quit()


if __name__ == "__main__":
    main()
