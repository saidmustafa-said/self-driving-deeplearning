# project/main.py
import torch
from track_environment import BallEnvironment
from model import ModelManager
import sys

import pygame  # Import pygame here in main.py as well


def run_episode(env, model_manager, training=True):
    """Run a single episode and update the model if training is True."""
    state = env.get_observation()
    done = False
    total_reward = 0

    while not done:
        # Event handling to ensure Pygame can process events, preventing freezes
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get the action from the model
        action = model_manager.choose_action(state)
        env.update_position(action[0], action[1])
        env.update_total_score()

        # Get the next state and reward, check for done
        next_state = env.get_observation()
        reward = next_state[7]  # Reward from observation vector
        done = env.check_collision()

        # Store experience and replay if in training mode
        if training:
            model_manager.store_experience(
                state, action, reward, next_state, done)
            model_manager.replay()

        state = next_state
        total_reward += reward

        # Render and update display
        env.render()
        pygame.display.flip()  # Ensure display updates every frame
        pygame.time.delay(10)  # Delay for smoother rendering

    env.reset()
    return total_reward


def train_model(num_episodes=1000):
    """Train the model over a set number of episodes."""
    env = BallEnvironment()
    model_manager = ModelManager()

    try:
        for episode in range(num_episodes):
            total_reward = run_episode(env, model_manager, training=True)
            print(
                f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f}")

            # Log the reward to TensorBoard
            model_manager.writer.add_scalar(
                "Episode Reward", total_reward, episode)

            # Save model every 10 episodes for progress tracking
            if (episode + 1) % 10 == 0:
                torch.save(model_manager.model.state_dict(),
                           "trained_model.pth")
                print("Model checkpoint saved.")

        # Final save after training completes
        torch.save(model_manager.model.state_dict(), "trained_model.pth")
        print("Training complete and model saved.")

    except KeyboardInterrupt:
        print("Training interrupted. Saving model...")
        torch.save(model_manager.model.state_dict(), "trained_model.pth")
        print("Model saved after interruption.")



def evaluate_model(num_episodes=10):
    """Evaluate the model over a set number of episodes without training."""
    env = BallEnvironment()
    model_manager = ModelManager()
    model_manager.model.load_state_dict(torch.load("trained_model.pth"))

    for episode in range(num_episodes):
        total_reward = run_episode(env, model_manager, training=False)
        print(f"Evaluation Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    # Choose either to train or evaluate the model
    # train_model()  # Comment this line to evaluate instead
    evaluate_model()  # Uncomment this line to evaluate
