import pygame
import sys
import math

# Initialize Pygame
pygame.init()

# Define display dimensions and colors
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (50, 50, 50)
TRACK_COLOR = (200, 200, 200)
CAR_COLOR = (255, 0, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Track with Car Movement")

# Track boundaries (simple oval-shaped track for now)
track_center = (WIDTH // 2, HEIGHT // 2)
track_radius = 200

# Car properties
car_width, car_height = 20, 10
car_x, car_y = WIDTH // 2, HEIGHT // 2 - \
    track_radius + 10  # Start position on the track
car_speed = 0  # Initial speed
car_angle = 0  # Initial angle (0 degrees is pointing right)

# Car movement parameters
max_speed = 5
acceleration = 0.1
deceleration = 0.1
turn_speed = 3  # degrees per frame

# Game loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(BACKGROUND_COLOR)

    # Draw track (oval shape for simplicity)
    pygame.draw.circle(screen, TRACK_COLOR, track_center, track_radius, 40)

    # Draw the car as a rectangle
    car_rect = pygame.Rect(car_x - car_width // 2, car_y -
                           car_height // 2, car_width, car_height)
    pygame.draw.rect(screen, CAR_COLOR, car_rect)
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

    # Draw the car as a rotated rectangle
    car_rect = pygame.Rect(0, 0, car_width, car_height)
    car_rect.center = (car_x, car_y)
    rotated_car = pygame.transform.rotate(
        screen.subsurface(car_rect).copy(), -car_angle)
    screen.blit(rotated_car, rotated_car.get_rect(center=car_rect.center))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
