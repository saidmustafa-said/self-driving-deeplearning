import pygame
import sys

# Initialize Pygame
pygame.init()

# Define display dimensions and colors
WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (50, 50, 50)
TRACK_COLOR = (200, 200, 200)
CAR_COLOR = (255, 0, 0)

# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Track with Car")

# Track boundaries (simple oval-shaped track for now)
track_center = (WIDTH // 2, HEIGHT // 2)
track_radius = 200

# Car properties
car_width, car_height = 20, 10
car_x, car_y = WIDTH // 2, HEIGHT // 2 - \
    track_radius + 10  # Start position on the track
car_speed = 0  # Initial speed
car_angle = 0  # Initial angle

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

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
