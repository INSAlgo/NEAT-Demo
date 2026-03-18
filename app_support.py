import glob
import math
import os
import pickle
import random

import neat
import pygame

from ai_core import (
    CarAI,
    NEATBrain,
    QLearningBrain,
    SIM_WIDTH,
    WINDOW_HEIGHT,
    CAR_SIZE,
    SENSOR_RANGE,
    MAX_FRAMES_PER_GEN,
)

CACHE_DIR = os.path.join(os.path.dirname(__file__), '__model_cache__')
os.makedirs(CACHE_DIR, exist_ok=True)

WINDOW_WIDTH = 1300

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
PANEL_COLOR = (230, 230, 240)
BLUE = (50, 50, 255)
GREEN = (50, 255, 50)
CYAN = (0, 255, 255)


def get_cache_filename(iterations):
    return os.path.join(CACHE_DIR, f"best_model_iter_{iterations}.pkl")


def get_latest_checkpoint():
    checkpoint_pattern = os.path.join(CACHE_DIR, 'neat-checkpoint-*')
    checkpoints = glob.glob(checkpoint_pattern)
    if not checkpoints:
        return None

    def checkpoint_gen(path):
        try:
            return int(os.path.basename(path).split('-')[-1])
        except Exception:
            return -1

    checkpoints.sort(key=checkpoint_gen)
    return checkpoints[-1]


def get_valid_target_pos(obstacles):
    target_radius = 20
    while True:
        tx = random.randint(50, SIM_WIDTH - 50)
        ty = random.randint(50, WINDOW_HEIGHT - 50)
        hit = False
        for rx, ry, rw, rh in obstacles:
            if rx - target_radius <= tx <= rx + rw + target_radius and ry - target_radius <= ty <= ry + rh + target_radius:
                hit = True
                break
        if not hit:
            return (tx, ty)


def generate_random_environment():
    obstacles = []
    for _ in range(8):
        w = random.randint(40, 100)
        h = random.randint(40, 100)
        x = random.randint(150, SIM_WIDTH - 150)
        y = random.randint(50, WINDOW_HEIGHT - 150)
        obstacles.append((x, y, w, h))

    start_pos = (random.randint(50, 150), random.randint(50, WINDOW_HEIGHT - 50))
    targets = [get_valid_target_pos(obstacles) for _ in range(5)]
    return obstacles, start_pos, targets


def train_headless(target_iterations):
    config_path = os.path.join(os.path.dirname(__file__), 'config-feedforward.txt')
    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
    )

    cache_file = get_cache_filename(target_iterations)
    if os.path.exists(cache_file):
        print(f"Model for {target_iterations} iterations already exists. Done.")
        return

    checkpoint = get_latest_checkpoint()
    if checkpoint:
        print(f"Restoring from checkpoint: {checkpoint}")
        try:
            population = neat.Checkpointer.restore_checkpoint(checkpoint)
        except Exception as exc:
            print(f"Checkpoint load failed ({exc}). Starting a fresh population instead.")
            population = neat.Population(config)
    else:
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join(CACHE_DIR, 'neat-checkpoint-')))

    obstacles, start_pos, targets = generate_random_environment()

    def eval_genomes(genomes, config_instance):
        nonlocal obstacles, start_pos, targets
        cars = []
        for _, genome in genomes:
            genome.fitness = 0
            brain = NEATBrain.create(genome, config_instance)
            cars.append(CarAI(start_pos[0], start_pos[1], genome, brain))

        if random.random() < 0.2:
            obstacles, start_pos, targets = generate_random_environment()

        frames = 0
        while frames < MAX_FRAMES_PER_GEN and any(c.alive for c in cars):
            for car in cars:
                if not car.alive:
                    continue
                if car.target_idx < len(targets):
                    target_pos = targets[car.target_idx]
                    steering, speed = car.think(target_pos)
                    car.update(steering, speed, obstacles, target_pos)
                    if car.reached_target:
                        car.reached_target = False
                    if car.genome is not None:
                        car.genome.fitness -= 0.1
                else:
                    car.alive = False
            frames += 1

        for car in cars:
            target_pos = targets[min(car.target_idx, len(targets) - 1)]
            dx = target_pos[0] - car.x
            dy = target_pos[1] - car.y
            dist = math.sqrt(dx * dx + dy * dy)
            if car.genome is not None:
                car.genome.fitness += car.fitness
                car.genome.fitness += (SIM_WIDTH - dist) / 10

    current_gen = population.generation if hasattr(population, 'generation') else 0
    if target_iterations > current_gen:
        best_genome = population.run(eval_genomes, target_iterations - current_gen)
        print(f"Training complete! Best fitness: {best_genome.fitness}")
        with open(get_cache_filename(target_iterations), 'wb') as f:
            pickle.dump(best_genome, f)
    else:
        print("Target iterations already matched or exceeded by current checkpoint.")


class Button:
    def __init__(self, x, y, w, h, text, on_click):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.on_click = on_click
        self.hovered = False

    def draw(self, surface, font):
        color = GRAY if not self.hovered else WHITE
        pygame.draw.rect(surface, color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        text_surf = font.render(self.text, True, BLACK)
        surface.blit(text_surf, (self.rect.x + (self.rect.width - text_surf.get_width()) // 2,
                                 self.rect.y + (self.rect.height - text_surf.get_height()) // 2))


def draw_car_ai(surface, car, color=CYAN, draw_sensors=False):
    car_rect = pygame.Rect(0, 0, CAR_SIZE, CAR_SIZE)
    car_rect.center = (int(car.x), int(car.y))

    angle_degrees = -math.degrees(car.angle)
    rotated_surface = pygame.Surface((CAR_SIZE, CAR_SIZE), pygame.SRCALPHA)
    pygame.draw.rect(rotated_surface, color, rotated_surface.get_rect())
    pygame.draw.line(rotated_surface, BLACK, (CAR_SIZE // 2, CAR_SIZE // 2), (CAR_SIZE, CAR_SIZE // 2), 3)

    rotated_surface = pygame.transform.rotate(rotated_surface, angle_degrees)
    rotated_rect = rotated_surface.get_rect(center=car_rect.center)
    surface.blit(rotated_surface, rotated_rect)

    if draw_sensors and car.alive:
        angles = [car.angle - math.pi / 4, car.angle - math.pi / 8, car.angle, car.angle + math.pi / 8, car.angle + math.pi / 4]
        for i, ray_angle in enumerate(angles):
            length = car.sensor_readings[i] * SENSOR_RANGE
            ex = car.x + math.cos(ray_angle) * length
            ey = car.y + math.sin(ray_angle) * length
            pygame.draw.line(surface, GREEN, (int(car.x), int(car.y)), (int(ex), int(ey)), 1)
