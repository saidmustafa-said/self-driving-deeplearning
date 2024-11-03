# track_environment.py

import pygame
import math

# Initialize Pygame
pygame.init()

# Screen setup
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (50, 50, 50)
TRACK_COLOR = (200, 200, 200)
WALL_COLOR = (100, 100, 100)
CAR_COLOR = (255, 0, 0)

# Track and Car Properties
track_center = (WIDTH // 2, HEIGHT // 2)
outer_radius = 200
inner_radius = 160

# Car properties
car_width, car_height = 20, 10
max_speed = 5
acceleration = 0.1
deceleration = 0.1
turn_speed = 3


class CarEnvironment:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("2D Track with Car Movement")
        self.car_x, self.car_y = WIDTH // 2, HEIGHT // 2 - outer_radius + 10
        self.car_speed = 0
        self.car_angle = 0

    def reset(self):
        """Reset the car to the starting position."""
        self.car_x, self.car_y = WIDTH // 2, HEIGHT // 2 - outer_radius + 10
        self.car_speed = 0
        self.car_angle = 0

    def update_position(self, acceleration, turn_angle):
        """Update the car's position based on acceleration and turn angle."""
        self.car_speed = min(
            max_speed, max(-max_speed, self.car_speed + acceleration))
        self.car_angle += turn_angle * turn_speed

        # Update position based on angle and speed
        radian_angle = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.cos(radian_angle)
        self.car_y -= self.car_speed * math.sin(radian_angle)

    def check_collision(self):
        """Check if the car is out of track bounds."""
        distance_from_center = math.sqrt(
            (self.car_x - track_center[0]) ** 2 + (self.car_y - track_center[1]) ** 2)
        return not (inner_radius < distance_from_center < outer_radius)

    def render(self, total_reward):
        """Render the track, car, and score."""
        self.screen.fill(BACKGROUND_COLOR)
        pygame.draw.circle(self.screen, TRACK_COLOR,
                           track_center, outer_radius, 40)
        pygame.draw.circle(self.screen, WALL_COLOR,
                           track_center, outer_radius, 5)
        pygame.draw.circle(self.screen, WALL_COLOR,
                           track_center, inner_radius, 5)

        # Draw the car
        car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
        car_surface.fill(CAR_COLOR)
        rotated_car = pygame.transform.rotate(car_surface, -self.car_angle)
        rotated_rect = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, rotated_rect.topleft)

        # Display reward
        font = pygame.font.Font(None, 36)
        reward_text = font.render(
            f"Total Reward: {total_reward:.2f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (10, 10))

        pygame.display.flip()
