# main.py
import torch
from track_environment import BallEnvironment
from model import ModelManager


def run_episode(env, model_manager, training=True):
    """Run a single episode and update the model if training is True."""
    state = env.get_observation()
    done = False
    total_reward = 0

    while not done:
        action = model_manager.choose_action(state)
        env.update_position(action[0], action[1])
        env.update_total_score()

        next_state = env.get_observation()
        reward = next_state[7]  # Reward from observation vector
        done = env.check_collision()

        if training:
            model_manager.store_experience(
                state, action, reward, next_state, done)
            model_manager.replay()

        state = next_state
        total_reward += reward
        env.render()

    env.reset()
    return total_reward


def train_model(num_episodes=1000):
    """Train the model over a set number of episodes."""
    env = BallEnvironment()
    model_manager = ModelManager()

    for episode in range(num_episodes):
        total_reward = run_episode(env, model_manager, training=True)
        print(
            f"Episode {episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f}")

        # Log the reward to TensorBoard
        model_manager.writer.add_scalar(
            "Episode Reward", total_reward, episode)

    # Save the trained model
    torch.save(model_manager.model.state_dict(), "trained_model.pth")
    print("Training complete and model saved.")


def evaluate_model(num_episodes=10):
    """Evaluate the model over a set number of episodes without training."""
    env = BallEnvironment()
    model_manager = ModelManager()
    model_manager.model.load_state_dict(torch.load("trained_model.pth"))

    for episode in range(num_episodes):
        total_reward = run_episode(env, model_manager, training=False)
        print(f"Evaluation Episode {
              episode + 1}/{num_episodes} - Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    # Choose either to train or evaluate the model
    train_model()  # Comment this line to evaluate instead
    # evaluate_model()  # Uncomment this line to evaluate
