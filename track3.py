import torch.optim as optim
import torch.nn as nn
import torch
import pygame
import sys
import math
import random

# Initialize Pygame
pygame.init()

# Define display dimensions and colors
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (50, 50, 50)
TRACK_COLOR = (200, 200, 200)
WALL_COLOR = (100, 100, 100)
CAR_COLOR = (255, 0, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Track with Car Movement")

# Track boundaries (simple oval-shaped track for now)
track_center = (WIDTH // 2, HEIGHT // 2)
outer_radius = 200
inner_radius = 160

# Car properties
car_width, car_height = 20, 10
car_x, car_y = WIDTH // 2, HEIGHT // 2 - \
    outer_radius + 10  # Start position on the track
car_speed = 0  # Initial speed
car_angle = 0  # Initial angle (0 degrees is pointing right)

# Car movement parameters
max_speed = 5
acceleration = 0.1
deceleration = 0.1
turn_speed = 3  # degrees per frame

# Reward parameters
reward = 0
total_reward = 0

# Reset car position to a random point on the track


def reset_car():
    global car_x, car_y, car_speed, car_angle, total_reward
    car_x, car_y = WIDTH // 2, HEIGHT // 2 - outer_radius + 10
    car_speed = 0
    car_angle = 0
    total_reward = 0  # Reset reward at the start of each episode


# Game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BACKGROUND_COLOR)

    # Draw track walls
    pygame.draw.circle(screen, TRACK_COLOR, track_center, outer_radius, 40)
    pygame.draw.circle(screen, WALL_COLOR, track_center,
                       outer_radius, 5)  # Outer wall
    pygame.draw.circle(screen, WALL_COLOR, track_center,
                       inner_radius, 5)  # Inner wall

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Car movement controls
    keys = pygame.key.get_pressed()

    # Speed control
    if keys[pygame.K_UP]:  # Accelerate
        car_speed = min(car_speed + acceleration, max_speed)
    elif keys[pygame.K_DOWN]:  # Decelerate
        car_speed = max(car_speed - deceleration, -max_speed)
    else:  # Natural deceleration when no key is pressed
        if car_speed > 0:
            car_speed = max(car_speed - deceleration, 0)
        elif car_speed < 0:
            car_speed = min(car_speed + deceleration, 0)

    # Turning control
    if keys[pygame.K_LEFT]:
        car_angle += turn_speed
    if keys[pygame.K_RIGHT]:
        car_angle -= turn_speed

    # Convert car angle to radians for movement calculation
    radian_angle = math.radians(car_angle)

    # Update car position based on speed and angle
    car_x += car_speed * math.cos(radian_angle)
    car_y -= car_speed * math.sin(radian_angle)

    # Calculate reward based on distance moved
    reward = car_speed * 0.1  # Reward for moving forward

    # Collision detection with track walls
    distance_from_center = math.sqrt(
        (car_x - track_center[0]) ** 2 + (car_y - track_center[1]) ** 2)
    if distance_from_center > outer_radius or distance_from_center < inner_radius:
        reward = -10  # Penalty for crashing
        reset_car()  # Reset car position after crash

    # Accumulate total reward
    total_reward += reward

    # Draw the car as a rotated rectangle
    car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
    car_surface.fill(CAR_COLOR)
    rotated_car = pygame.transform.rotate(car_surface, -car_angle)
    rotated_rect = rotated_car.get_rect(center=(car_x, car_y))
    screen.blit(rotated_car, rotated_rect.topleft)

    # Display reward information
    font = pygame.font.Font(None, 36)
    reward_text = font.render(
        f"Reward: {total_reward:.2f}", True, (255, 255, 255))
    screen.blit(reward_text, (10, 10))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()


# Neural Network Model

class CarNet(nn.Module):
    def __init__(self):
        super(CarNet, self).__init__()
        # Input layer: takes in car_x, car_y, car_angle, car_speed
        # Output layer: predicts actions (e.g., turn, accelerate, decelerate)
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 outputs: acceleration and turn angle

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Outputs raw values for actions
        return x


# Initialize model, optimizer, and loss function
model = CarNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Example function to take an action based on the model's output


def get_action(observation):
    with torch.no_grad():
        output = model(torch.tensor(observation, dtype=torch.float32))
        # Positive values for accelerate, negative for decelerate
        acceleration = output[0].item()
        turn_angle = output[1].item()  # Positive for left, negative for right
        return acceleration, turn_angle


# Example observation (x position, y position, car angle, car speed)
observation = [car_x, car_y, car_angle, car_speed]
acceleration, turn_angle = get_action(observation)
print("Acceleration:", acceleration, "Turn Angle:", turn_angle)


