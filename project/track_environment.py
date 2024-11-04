# project/track_environment.py
import sys
import pygame
import math

# Screen settings
WIDTH = 800
HEIGHT = 700
BACKGROUND_COLOR = (50, 50, 50)

# Track settings
TRACK_COLOR = (200, 200, 200)
WALL_COLOR = (100, 100, 100)
TRACK_WIDTH = 150  # Width of the track

# Ball settings
BALL_COLOR = (255, 0, 0)
BALL_RADIUS = 10
MAX_SPEED = 5  # Maximum speed of the ball
ACCELERATION = 0.2  # Acceleration when moving
DRAG = 0.05  # Drag to slow down the ball more naturally


class BallEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Curved S-Shaped Track with Ball Movement")
        self.reset()
        self.total_reward = 0  # Keep track of total reward for debugging purposes

    def reset(self):
        """Reset the ball to the starting position and reset cumulative rewards and penalties."""
        self.ball_x = WIDTH // 2  # Start in the middle of the screen width
        self.ball_y = HEIGHT - BALL_RADIUS  # Start at the bottom of the screen
        self.ball_speed = 0  # Initial speed
        self.ball_direction = 0  # Initial direction
        self.cumulative_reward = 0  # Reset cumulative reward
        self.cumulative_penalty = 0  # Reset cumulative penalty
        self.total_score = 0  # Reset total score for the episode

    def update_position(self, accel_input, turn_input):
        """Update ball position based on acceleration and direction."""
        # Increase speed if acceleration input is provided
        if accel_input > 0:
            if self.ball_speed < MAX_SPEED:
                self.ball_speed += ACCELERATION
        else:
            self.ball_speed *= (1 - DRAG)

        # Calculate the turning angle using turn_input * 50
        angle = math.radians(turn_input * 50)
        self.ball_direction = turn_input  # Keep the original input for relative heading

        # Calculate vertical and horizontal speeds
        vertical_speed = self.ball_speed * math.cos(angle)
        horizontal_speed = self.ball_speed * math.sin(angle)
        self.ball_speed = min(self.ball_speed, MAX_SPEED)

        # Update ball's position
        self.ball_y -= vertical_speed
        self.ball_x += horizontal_speed

        # Bound the ball within the track limits
        left_bound = self.get_track_left(self.ball_y)
        right_bound = self.get_track_right(self.ball_y)
        self.ball_x = max(left_bound, min(self.ball_x, right_bound))

    def get_observation(self):
        """Get a detailed observation vector representing the current environment state."""
        left_bound = self.get_track_left(self.ball_y)
        right_bound = self.get_track_right(self.ball_y)

        # Normalized ball position on the screen
        normalized_x = self.ball_x / WIDTH
        normalized_y = self.ball_y / HEIGHT

        # Distance to left and right boundaries, normalized by track width
        track_width = right_bound - left_bound
        distance_to_left = (self.ball_x - left_bound) / track_width
        distance_to_right = (right_bound - self.ball_x) / track_width

        # Use original turn_input value for relative heading
        relative_heading = self.ball_direction

        # Normalized speed components
        angle_rad = math.radians(self.ball_direction * 50)
        horizontal_speed = (self.ball_speed * math.sin(angle_rad)
                            ) / MAX_SPEED  # Normalized -1 to 1
        vertical_speed = (self.ball_speed * math.cos(angle_rad)
                          ) / MAX_SPEED    # Normalized 0 to 1

        observation = [
            round(normalized_x, 5),                 # Normalized x-position
            round(normalized_y, 5),                 # Normalized y-position
            # Heading relative to track centerline
            round(relative_heading, 5),
            # Normalized horizontal speed (-1 to 1)
            round(horizontal_speed, 5),
            # Normalized vertical speed (0 to 1)
            round(vertical_speed, 5),
            # Distance to left boundary (0 to 1)
            round(distance_to_left, 5),
            # Distance to right boundary (0 to 1)
            round(distance_to_right, 5),
            # Speed-based reward
            round(self.calculate_reward(normalized_y), 5),
            round(self.calculate_penalty(), 5),     # Collision penalty
            round(self.total_score, 5)              # Total score
        ]
        return observation

    def get_track_left(self, y):
        """Calculate the left boundary of the S-shaped track based on y position."""
        return (WIDTH / 2 - TRACK_WIDTH / 2) + TRACK_WIDTH * math.sin(2 * math.pi * (y / HEIGHT))

    def get_track_right(self, y):
        """Calculate the right boundary of the S-shaped track based on y position."""
        return (WIDTH / 2 + TRACK_WIDTH / 2) + TRACK_WIDTH * math.sin(2 * math.pi * (y / HEIGHT))

    def check_collision(self):
        """Check if the ball has collided with the sides or reached the top."""
        ball_left = self.ball_x - BALL_RADIUS
        ball_right = self.ball_x + BALL_RADIUS
        left_bound = self.get_track_left(self.ball_y)
        right_bound = self.get_track_right(self.ball_y)

        if ball_left <= left_bound or ball_right >= right_bound:
            return True

        if self.ball_y <= BALL_RADIUS:
            print("Reached the finish line!")
            return True

        return False

    def calculate_reward(self, normalized_y):
        """Calculate reward based on speed and Y position and accumulate it."""
        speed_reward = self.ball_speed * (1 - normalized_y)
        self.cumulative_reward += speed_reward  # Accumulate reward
        return speed_reward

    def calculate_penalty(self):
        """Apply a penalty if there’s a collision and accumulate it."""
        collision = self.check_collision()
        if collision:
            penalty = -10
            self.cumulative_penalty += penalty  # Accumulate penalty
            return penalty
        return 0

    def update_total_score(self):
        """Calculate the total score based on cumulative reward and penalty."""
        self.total_score = self.cumulative_reward + self.cumulative_penalty

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.render_track()
        self.render_ball()
        self.display_info()
        pygame.display.flip()

    def render_track(self):
        for y in range(0, HEIGHT, 5):
            x_left = self.get_track_left(y)
            x_right = self.get_track_right(y)
            pygame.draw.line(self.screen, TRACK_COLOR,
                             (x_left, y), (x_right, y))
            pygame.draw.line(self.screen, WALL_COLOR,
                             (x_left, y), (x_left, y), 5)
            pygame.draw.line(self.screen, WALL_COLOR,
                             (x_right, y), (x_right, y), 5)

    def render_ball(self):
        pygame.draw.circle(self.screen, BALL_COLOR, (int(
            self.ball_x), int(self.ball_y)), BALL_RADIUS)

    def display_info(self):
        """Display the observation values on the screen for debugging."""
        font = pygame.font.Font(None, 24)
        observation = self.get_observation()
        labels = [
            "Normalized X Position:", "Normalized Y Position:", "Relative Heading:",
            "Horizontal Speed:", "Normalized Vertical Speed:", "Distance to Left Boundary:",
            "Distance to Right Boundary:", "Speed-Based Reward:", "Collision Penalty:", "Total Score:"
        ]
        for i, (label, value) in enumerate(zip(labels, observation)):
            info_text = f"{label} {value:.2f}"
            text_surface = font.render(info_text, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, 10 + i * 30))

    def handle_controls(self):
        keys = pygame.key.get_pressed()
        accel_input = 1 if keys[pygame.K_UP] else 0
        turn_input = - \
            1 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else 0)
        return accel_input, turn_input


def main():
    env = BallEnvironment()
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        accel_input, turn_input = env.handle_controls()
        env.update_position(accel_input, turn_input)
        env.update_total_score()

        if env.check_collision():
            print(f"Collision or finish line. Total Score: {env.total_score}")
            observation = env.get_observation()
            print(f"Observation: {observation}")
            env.reset()

        env.render()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
