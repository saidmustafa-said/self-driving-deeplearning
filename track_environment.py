import pygame
import math

# Screen settings
WIDTH = 800
HEIGHT = 700
BACKGROUND_COLOR = (50, 50, 50)

# Track settings
TRACK_COLOR = (200, 200, 200)
WALL_COLOR = (100, 100, 100)
TRACK_WIDTH = 100  # Width of the track

# Ball settings
BALL_COLOR = (255, 0, 0)
BALL_RADIUS = 10
MAX_SPEED = 5  # Maximum speed of the ball
ACCELERATION = 0.2  # Acceleration when moving
DRAG = 0.1  # Drag to slow down the ball


class BallEnvironment:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Curved S-Shaped Track with Ball Movement")
        self.reset()

    def reset(self):
        """Reset the ball to the starting position."""
        self.ball_x = WIDTH // 2  # Start in the middle of the screen width
        self.ball_y = HEIGHT - BALL_RADIUS  # Start at the bottom of the screen
        self.ball_speed = 0  # Initial speed

    def update_position(self, accel_input, turn_input):
        """Update ball position based on acceleration and direction."""
        # Apply acceleration
        if accel_input > 0:
            if self.ball_speed < MAX_SPEED:
                self.ball_speed += ACCELERATION
        else:
            # Apply drag when no acceleration
            self.ball_speed *= (1 - DRAG)

        # Update vertical position
        self.ball_y -= self.ball_speed  # Move upward based on speed

        # Update horizontal position based on turning
        self.ball_x += turn_input

        # Clamp horizontal position to stay within track boundaries
        left_bound = self.get_track_left(self.ball_y)
        right_bound = self.get_track_right(self.ball_y)
        self.ball_x = max(left_bound, min(self.ball_x, right_bound))

    def get_track_left(self, y):
        """Calculate the left boundary of the S-shaped track based on y position."""
        # Increase the frequency and amplitude for more curvature
        return (WIDTH / 2 - TRACK_WIDTH / 2) + (TRACK_WIDTH / 1) * math.sin(5 * math.pi * (y / HEIGHT))

    def get_track_right(self, y):
        """Calculate the right boundary of the S-shaped track based on y position."""
        # Increase the frequency and amplitude for more curvature
        return (WIDTH / 2 + TRACK_WIDTH / 2) + (TRACK_WIDTH / 1) * math.sin(5 * math.pi * (y / HEIGHT))

    def check_collision(self):
        """Check if the ball has collided with the sides or reached the top."""
        # Check for collision with the track boundaries
        ball_left = self.ball_x - BALL_RADIUS
        ball_right = self.ball_x + BALL_RADIUS
        left_bound = self.get_track_left(self.ball_y)
        right_bound = self.get_track_right(self.ball_y)
        if ball_left <= left_bound:
            print("Left collision detected!")
            return True  # Collision with sides
        if ball_right >= right_bound:
            print("Right collision detected!")
            return True  # Collision with sides
        if self.ball_y < 0:  # Ball has reached the top of the screen
            print("Winning condition reached!")
            return True  # Winning condition
        return False  # No collision

    def render(self, total_reward):
        """Render the track, ball, and reward display."""
        self.screen.fill(BACKGROUND_COLOR)

        # Draw the S-shaped track using lines
        for y in range(0, HEIGHT):
            x_left = self.get_track_left(y)
            x_right = self.get_track_right(y)
            pygame.draw.line(self.screen, TRACK_COLOR,
                             (x_left, y), (x_right, y))

        # Draw walls
        for y in range(0, HEIGHT, 5):
            pygame.draw.line(self.screen, WALL_COLOR, (self.get_track_left(
                y), y), (self.get_track_left(y), y), 5)
            pygame.draw.line(self.screen, WALL_COLOR, (self.get_track_right(
                y), y), (self.get_track_right(y), y), 5)

        # Draw the ball
        pygame.draw.circle(self.screen, BALL_COLOR, (int(
            self.ball_x), int(self.ball_y)), BALL_RADIUS)

        # Display reward
        font = pygame.font.Font(None, 36)
        reward_text = font.render(
            f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (10, 10))

        pygame.display.flip()


def main():
    env = BallEnvironment()
    clock = pygame.time.Clock()
    total_reward = 0
    running = True

    while running:
        accel_input = 0
        turn_input = 0

        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get key states for movement control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:       # Accelerate
            accel_input = 1
        if keys[pygame.K_LEFT]:     # Turn left
            turn_input = -2  # Move left
        if keys[pygame.K_RIGHT]:    # Turn right
            turn_input = 2   # Move right

        # Update ball position and check for collisions
        env.update_position(accel_input, turn_input)
        if env.check_collision():
            print("Collision detected or top reached! Restarting.")
            env.reset()  # Reset ball position
            total_reward = 0  # Reset reward on collision or winning
        else:
            total_reward += 1  # Increase reward for staying on track

        # Render the environment
        env.render(total_reward)

        # Cap the frame rate
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
