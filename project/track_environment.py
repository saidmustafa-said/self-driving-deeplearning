import pygame
import math

# Screen settings
WIDTH = 800
HEIGHT = 600
BACKGROUND_COLOR = (50, 50, 50)

# Track settings
TRACK_COLOR = (200, 200, 200)
TRACK_WIDTH = 100  # Width of the track

# Ball settings
BALL_COLOR = (255, 0, 0)
BALL_RADIUS = 10
MAX_SPEED = 5  # Maximum speed of the ball
ACCELERATION = 0.2  # Acceleration when moving
DRAG = 0.1  # Drag to slow down the ball


class Ball:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = 0
        self.acceleration = 0
        self.turn = 0

    def move(self):
        # Update speed based on acceleration
        self.speed += self.acceleration
        self.speed = max(0, min(self.speed, MAX_SPEED))  # Clamp speed

        # Move the ball based on speed
        self.x += self.turn
        self.y -= self.speed  # Move up the screen

        # Keep the ball within the track boundaries
        if self.x < BALL_RADIUS:
            self.x = BALL_RADIUS
        elif self.x > WIDTH - BALL_RADIUS:
            self.x = WIDTH - BALL_RADIUS

    def draw(self, screen):
        pygame.draw.circle(screen, BALL_COLOR,
                           (int(self.x), int(self.y)), BALL_RADIUS)


class BallEnvironment:
    def __init__(self, num_balls=5):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Curved S-Shaped Track with Multiple Balls")
        self.num_balls = num_balls
        self.balls = [Ball(WIDTH // 2, HEIGHT - BALL_RADIUS)
                      for _ in range(num_balls)]

    def reset(self):
        """Reset all balls to their starting positions."""
        self.balls = [Ball(WIDTH // 2, HEIGHT - BALL_RADIUS)
                      for _ in range(self.num_balls)]

    def check_collisions(self):
        """Check if any ball has collided with the sides or reached the top."""
        for ball in self.balls:
            if ball.y < 0:  # Winning condition
                return True
            if ball.x < BALL_RADIUS or ball.x > WIDTH - BALL_RADIUS:
                return True  # Collision with sides
        return False

    def render(self):
        """Render the balls and the background."""
        self.screen.fill(BACKGROUND_COLOR)

        # Draw the S-shaped track
        for y in range(0, HEIGHT):
            x_left = self.get_track_left(y)
            x_right = self.get_track_right(y)
            pygame.draw.line(self.screen, TRACK_COLOR,
                             (x_left, y), (x_right, y))

        # Draw the balls
        for ball in self.balls:
            ball.draw(self.screen)

        pygame.display.flip()

    def get_track_left(self, y):
        """Calculate the left boundary of the S-shaped track based on y position."""
        return (WIDTH / 2 - TRACK_WIDTH / 2) + (TRACK_WIDTH / 1) * math.sin(5 * math.pi * (y / HEIGHT))

    def get_track_right(self, y):
        """Calculate the right boundary of the S-shaped track based on y position."""
        return (WIDTH / 2 + TRACK_WIDTH / 2) + (TRACK_WIDTH / 1) * math.sin(5 * math.pi * (y / HEIGHT))
