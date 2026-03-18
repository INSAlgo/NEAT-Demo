# ============================================================================
# ENTRY SCRIPT & PYGAME FRONTEND
# ============================================================================
import os
import sys
import pickle
import math
import argparse
from typing import Any

import neat

# CLI arguments must be parsed before initializing pygame to avoid an unneeded window popup on pure headless servers
parser = argparse.ArgumentParser(description="Educational Car AI using NEAT")
parser.add_argument("--cli", action="store_true", help="Run training purely in terminal without UI")
parser.add_argument("--iters", type=int, default=50, help="Number of iterations to train (in CLI mode)")
args = parser.parse_args()

# Import the core AI backend
from ai_core import CarAI, NEATBrain, QLearningBrain, SIM_WIDTH, WINDOW_HEIGHT
from app_support import (
    Button,
    CACHE_DIR,
    CAR_SIZE,
    CYAN,
    DARK_GRAY,
    BLACK,
    BLUE,
    GRAY,
    GREEN,
    MAX_FRAMES_PER_GEN,
    PANEL_COLOR,
    SENSOR_RANGE,
    WHITE,
    WINDOW_WIDTH,
    draw_car_ai,
    generate_random_environment,
    get_cache_filename,
    get_latest_checkpoint,
    train_headless,
)

if args.cli:
    print(f"Starting headless CLI training for {args.iters} iterations...")
    train_headless(args.iters)
    sys.exit(0)

# If not running in CLI mode, import PyGame and build visual app
import pygame


class RLAppUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("RL Educational Simulation - Learn AI Driving")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 32)
        
        self.state = "IDLE"
        self.iterations_to_train: int = 10
        self.current_generation: int = 0
        
        # Sim variables
        self.obstacles = []
        self.start_pos = (100, WINDOW_HEIGHT//2)
        self.targets = [(SIM_WIDTH - 100, WINDOW_HEIGHT//2)]
        self.cars: list[CarAI] = []
        self.fitness_history: list[float] = []
        
        # Interaction variables
        self.dragging_target_idx = -1
        self.dragging_start = False
        self.drawing_obstacle_start = None
        self.mouse_current_pos = (0, 0)
        
        # Algorithm selection
        self.algorithms = ["NEAT", "Q-LEARNING"]
        self.current_algo_idx = 0
        self.q_table: dict[Any, list[float]] = {}
        self.enable_blocks = True
        
        self.create_buttons()
        
        config_path = os.path.join(os.path.dirname(__file__), 'config-feedforward.txt')
        self.config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction,
            neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
        )
        self.population = None
        self.best_genome_cache: Any | None = None

        self.randomize_environment()

    def cycle_algorithm(self):
        self.current_algo_idx = (self.current_algo_idx + 1) % len(self.algorithms)
        # Update button text
        self.buttons[0].text = f"Algorithm: {self.algorithms[self.current_algo_idx]}"
        self.reset_state()

    def toggle_blocks(self):
        self.enable_blocks = not self.enable_blocks
        self.buttons[2].text = f"Blocks: {'ON' if self.enable_blocks else 'OFF'}"

    def create_buttons(self):
        bx = SIM_WIDTH + 20
        bw = 360
        half_bw = 175
        self.buttons = [
            Button(bx, 60, bw, 30, f"Algorithm: {self.algorithms[self.current_algo_idx]}", self.cycle_algorithm),
            Button(bx, 95, bw, 30, "Randomize Environment", self.randomize_environment),
            Button(bx, 130, bw, 30, f"Blocks: {'ON' if self.enable_blocks else 'OFF'}", self.toggle_blocks),
            Button(bx, 165, half_bw, 30, "- Iterations", lambda: self.adjust_iters(-5)),
            Button(bx + 185, 165, half_bw, 30, "+ Iterations", lambda: self.adjust_iters(5)),
            Button(bx, 200, bw, 35, "Train Model", self.start_training),
            Button(bx, 240, bw, 35, "Preview Best Model", self.start_preview),
            Button(bx, 280, half_bw, 30, "Clear Cache", self.clear_cache),
            Button(bx + 185, 280, half_bw, 30, "Stop / Reset", self.reset_state)
        ]

    def adjust_iters(self, amt: int):
        self.iterations_to_train = max(5, self.iterations_to_train + amt)

    def randomize_environment(self):
        if self.state == "TRAINING": return
        # Using the helper from ai_core
        obs_tuples, self.start_pos, self.targets = generate_random_environment()
        self.obstacles = [pygame.Rect(*o) for o in obs_tuples]
        self.reset_cars_for_preview()

    def reset_state(self):
        self.state = "IDLE"
        self.cars = []

    def clear_cache(self):
        for filename in os.listdir(CACHE_DIR):
            file_path = os.path.join(CACHE_DIR, filename)
            try:
                if os.path.isfile(file_path): os.unlink(file_path)
            except Exception: pass
        self.population = None
        self.best_genome_cache = None
        self.current_generation = 0
        self.fitness_history = []
        self.reset_state()
        print("Cache and checkpoins cleared!")

    def load_cached_model(self):
        cache_file = get_cache_filename(self.iterations_to_train)
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def start_training(self):
        alg = self.algorithms[self.current_algo_idx]
        
        if alg == "NEAT":
            cached = self.load_cached_model()
            if cached:
                print("Loaded model from cache!")
                self.best_genome_cache = cached
                self.state = "IDLE"
                return
                
            if self.population is None:
                checkpoint = get_latest_checkpoint()
                if checkpoint:
                    print(f"Restoring from checkpoint: {checkpoint}")
                    self.population = neat.Checkpointer.restore_checkpoint(checkpoint)
                    self.current_generation = self.population.generation
                else:
                    self.population = neat.Population(self.config)
                    self.current_generation = 0
                    
                self.population.add_reporter(neat.Checkpointer(1, filename_prefix=os.path.join(CACHE_DIR, 'neat-checkpoint-')))
                
            if self.iterations_to_train <= self.current_generation:
                self.iterations_to_train = self.current_generation + 5
                
        elif alg == "Q-LEARNING":
            # Standard episodic RL iteration counter
            if self.current_generation >= self.iterations_to_train:
                self.iterations_to_train = self.current_generation + 5

        self.state = "TRAINING"

    def start_preview(self):
        alg = self.algorithms[self.current_algo_idx]
        if alg == "NEAT" and not self.best_genome_cache:
            self.start_training()
            return
        self.state = "PREVIEW"
        self.reset_cars_for_preview()

    def reset_cars_for_preview(self):
        alg = self.algorithms[self.current_algo_idx]
        if alg == "NEAT":
            if not self.best_genome_cache: return
            brain = NEATBrain.create(self.best_genome_cache, self.config)
            self.cars = [CarAI(self.start_pos[0], self.start_pos[1], None, brain)]
        elif alg == "Q-LEARNING":
            brain = QLearningBrain.create(self.q_table, epsilon=0.0) # Greedy for preview
            self.cars = [CarAI(self.start_pos[0], self.start_pos[1], None, brain)]

    def draw_graph(self):
        graph_rect = pygame.Rect(SIM_WIDTH + 20, 480, 360, 80)
        pygame.draw.rect(self.screen, WHITE, graph_rect)
        pygame.draw.rect(self.screen, DARK_GRAY, graph_rect, 1)
        
        title_surf = self.font.render("Fitness Over Generations", True, BLACK)
        self.screen.blit(title_surf, (graph_rect.x, graph_rect.y - 20))
        
        if len(self.fitness_history) < 2: return
            
        min_fit = min(self.fitness_history)
        max_fit = max(self.fitness_history)
        if max_fit == min_fit: max_fit += 1
            
        pts = []
        for i, fit in enumerate(self.fitness_history):
            x = graph_rect.x + (i / (len(self.fitness_history) - 1)) * graph_rect.width
            y = graph_rect.bottom - ((fit - min_fit) / (max_fit - min_fit)) * graph_rect.height
            pts.append((int(x), int(y)))
            
        if len(pts) >= 2:
            pygame.draw.lines(self.screen, BLUE, False, pts, 2)
            
        max_surf = self.font.render(f"{max_fit:.0f}", True, DARK_GRAY)
        min_surf = self.font.render(f"{min_fit:.0f}", True, DARK_GRAY)
        self.screen.blit(max_surf, (graph_rect.right - max_surf.get_width() - 5, graph_rect.top + 5))
        self.screen.blit(min_surf, (graph_rect.right - min_surf.get_width() - 5, graph_rect.bottom - 20))

    def draw_neat_network(self, genome: Any, config: Any, rect: pygame.Rect):
        """Draw a simplified graphical representation of the NEAT neural network."""
        inputs: list[int] = config.genome_config.input_keys  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        outputs: list[int] = config.genome_config.output_keys  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        hidden: list[int] = [k for k in genome.nodes.keys() if k not in outputs]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        node_pos: dict[int, tuple[float, float]] = {}
        # Left column (Inputs)
        if inputs:
            step_y = rect.height / (len(inputs) + 1)
            for i, k in enumerate(inputs):
                node_pos[k] = (rect.left + 15, rect.top + (i + 1) * step_y)
                
        # Right column (Outputs)
        if outputs:
            step_y = rect.height / (len(outputs) + 1)
            for i, k in enumerate(outputs):
                node_pos[k] = (rect.right - 15, rect.top + (i + 1) * step_y)
                
        # Middle column (Hidden)
        if hidden:
            step_y = rect.height / (len(hidden) + 1)
            for i, k in enumerate(hidden):
                node_pos[k] = (rect.left + rect.width / 2, rect.top + (i + 1) * step_y)
                
        # Draw connections
        for cg in genome.connections.values():
            if cg.enabled:
                in_node, out_node = cg.key
                if in_node in node_pos and out_node in node_pos:
                    color = (50, 200, 50) if cg.weight > 0 else (200, 50, 50)
                    width = max(1, min(4, int(abs(cg.weight))))
                    pygame.draw.line(self.screen, color, node_pos[in_node], node_pos[out_node], width)
                    
        # Draw nodes
        for k, pos in node_pos.items():
            color = BLUE if k in inputs else (GREEN if k in outputs else GRAY)
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), 5)
            pygame.draw.circle(self.screen, BLACK, (int(pos[0]), int(pos[1])), 5, 1)

    def draw_brain_preview(self):
        """Draw a visual representation of the 'brain' looking at the first car."""
        if not self.cars or self.state != "PREVIEW":
            return

        graph_rect = pygame.Rect(SIM_WIDTH + 20, 330, 360, 230)
        pygame.draw.rect(self.screen, WHITE, graph_rect)
        pygame.draw.rect(self.screen, DARK_GRAY, graph_rect, 1)

        title_surf = self.font.render("Live Brain View", True, BLACK)
        self.screen.blit(title_surf, (graph_rect.x, graph_rect.y - 20))

        car = self.cars[0]
        alg = self.algorithms[self.current_algo_idx]
        
        target = self.targets[min(car.target_idx, len(self.targets) - 1)]
        state = car.get_state(target)

        if alg == "NEAT":
            # Display sensor inputs and steering outputs simply
            y_offset = graph_rect.y + 10
            
            # Map input names to state array
            inputs = ["T-Angle", "T-Dist", "L-Wall", "FL-Wall", "F-Wall", "FR-Wall", "R-Wall"]
            for i, val in enumerate(state):
                label = self.font.render(f"{inputs[i]}: {val:.2f}", True, DARK_GRAY)
                self.screen.blit(label, (graph_rect.x + 10, y_offset + i * 28))
            
            # Outputs (ask the brain)
            if car.brain:
                steering, speed = car.brain.activate(state)
                y_offset = graph_rect.y + 10
                out_label = self.font.render(f"Steer: {steering:.2f}", True, BLUE)
                speed_label = self.font.render(f"Speed: {speed:.2f}", True, GREEN)
                self.screen.blit(out_label, (graph_rect.right - 120, y_offset))
                self.screen.blit(speed_label, (graph_rect.right - 120, y_offset + 20))
                
                # Render the network topology graph
                if car.genome:
                    net_rect = pygame.Rect(graph_rect.right - 180, graph_rect.y + 60, 160, 150)
                    self.draw_neat_network(car.genome, self.config, net_rect)

        elif alg == "Q-LEARNING":
            if isinstance(car.brain, QLearningBrain):
                q_state = car.brain.discretize(state)
                y_offset = graph_rect.y + 10
                
                state_label = self.font.render(f"Discrete State:", True, DARK_GRAY)
                state_val = self.font.render(f"{q_state}", True, BLUE)
                self.screen.blit(state_label, (graph_rect.x + 10, y_offset))
                self.screen.blit(state_val, (graph_rect.x + 10, y_offset + 20))

                known_states_label = self.font.render(f"Q-Table Size: {len(car.brain.q_table)}", True, DARK_GRAY)
                self.screen.blit(known_states_label, (graph_rect.x + 10, y_offset + 50))
                
                if q_state in car.brain.q_table:
                    vals = car.brain.q_table[q_state]
                    best_act = vals.index(max(vals))
                    best_act_label = self.font.render(f"Best Act: {best_act}", True, GREEN)
                    self.screen.blit(best_act_label, (graph_rect.x + 10, y_offset + 70))

    def draw_panel(self):
        pygame.draw.rect(self.screen, PANEL_COLOR, (SIM_WIDTH, 0, WINDOW_WIDTH - SIM_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(self.screen, DARK_GRAY, (SIM_WIDTH, 0), (SIM_WIDTH, WINDOW_HEIGHT), 2)
        
        title = self.large_font.render("RL Control Panel", True, BLACK)
        self.screen.blit(title, (SIM_WIDTH + 20, 20))
        
        info = [
            f"State: {self.state}",
            f"Target Iters: {self.iterations_to_train}",
            f"Current Gen: {self.current_generation}"
        ]
        for i, text in enumerate(info):
            surf = self.font.render(text, True, DARK_GRAY)
            self.screen.blit(surf, (SIM_WIDTH + 20, 580 + i*20))

        instructions = [
            "IDLE Controls:",
            "Left-Click+Drag: Move Target/Start",
            "Left-Click+Drag Empty: Draw Obstacle",
            "Right-Click Obstacle: Remove"
        ]
        
        for i, text in enumerate(instructions):
            surf = self.font.render(text, True, DARK_GRAY)
            self.screen.blit(surf, (SIM_WIDTH + 220, 580 + i*20))
                
        for btn in self.buttons:
            btn.draw(self.screen, self.font)
            
        self.draw_graph()
        self.draw_brain_preview()

    def draw_sim(self):
        pygame.draw.rect(self.screen, WHITE, (0, 0, SIM_WIDTH, WINDOW_HEIGHT))
        
        # Grid
        for x in range(0, SIM_WIDTH, 50):
            pygame.draw.line(self.screen, (240, 240, 240), (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, 50):
            pygame.draw.line(self.screen, (240, 240, 240), (0, y), (SIM_WIDTH, y))

        # Start and Targets
        pygame.draw.circle(self.screen, BLUE, (int(self.start_pos[0]), int(self.start_pos[1])), 10)
        
        for i, target in enumerate(self.targets):
            pygame.draw.circle(self.screen, GREEN, (int(target[0]), int(target[1])), 15)
            # Draw number
            t_surf = self.font.render(str(i+1), True, BLACK)
            t_rect = t_surf.get_rect(center=(int(target[0]), int(target[1])))
            self.screen.blit(t_surf, t_rect)
        
        for obs in self.obstacles:
            pygame.draw.rect(self.screen, DARK_GRAY, obs)
            pygame.draw.rect(self.screen, BLACK, obs, 2)
            
        if self.drawing_obstacle_start:
            x = min(self.drawing_obstacle_start[0], self.mouse_current_pos[0])
            y = min(self.drawing_obstacle_start[1], self.mouse_current_pos[1])
            w = abs(self.drawing_obstacle_start[0] - self.mouse_current_pos[0])
            h = abs(self.drawing_obstacle_start[1] - self.mouse_current_pos[1])
            temp_rect = pygame.Rect(x, y, w, h)
            pygame.draw.rect(self.screen, GRAY, temp_rect)
            pygame.draw.rect(self.screen, BLACK, temp_rect, 2)
            
        for car in self.cars:
            draw_car_ai(self.screen, car, draw_sensors=(self.state == "PREVIEW"))

    def eval_genomes(self, genomes: list[tuple[int, Any]], config: neat.Config):
        self.cars = []
        for genome_id, genome in genomes:
            genome.fitness = 0
            brain = NEATBrain.create(genome, config)
            self.cars.append(CarAI(self.start_pos[0], self.start_pos[1], genome, brain))

        if self.current_generation % 2 == 0:
            self.randomize_environment()

        frames = 0
        obs_tuples = [(o.x, o.y, o.width, o.height) for o in self.obstacles] if self.enable_blocks else []
        
        while frames < MAX_FRAMES_PER_GEN and any(c.alive for c in self.cars):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            for car in self.cars:
                if car.alive:
                    if car.target_idx < len(self.targets):
                        t_pos = self.targets[car.target_idx]
                        st, sp = car.think(t_pos)
                        car.update(st, sp, obs_tuples, t_pos)
                        if car.reached_target:
                            car.reached_target = False
                        car.genome.fitness -= 0.1
                    else:
                        car.alive = False # Finished all targets
                    
            self.draw_sim()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(500)
            frames += 1

        for car in self.cars:
            t_pos = self.targets[min(car.target_idx, len(self.targets)-1)]
            dx = t_pos[0] - car.x
            dy = t_pos[1] - car.y
            dist = math.sqrt(dx*dx + dy*dy)
            car.genome.fitness += car.fitness
            car.genome.fitness += (SIM_WIDTH - dist) / 10
            
        best_gen_fitness = max((car.genome.fitness for car in self.cars), default=0)
        self.fitness_history.append(best_gen_fitness)

    def run(self):
        running = True
        while running:
            self.mouse_current_pos = pygame.mouse.get_pos()
            mouse_pos = self.mouse_current_pos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    for btn in self.buttons:
                        btn.hovered = btn.rect.collidepoint(mouse_pos)
                    
                    if self.dragging_target_idx != -1:
                        self.targets[self.dragging_target_idx] = (min(max(0, mouse_pos[0]), SIM_WIDTH), min(max(0, mouse_pos[1]), WINDOW_HEIGHT))
                    elif self.dragging_start:
                        self.start_pos = (min(max(0, mouse_pos[0]), SIM_WIDTH), min(max(0, mouse_pos[1]), WINDOW_HEIGHT))

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        btn_clicked = False
                        for btn in self.buttons:
                            if btn.hovered: 
                                btn.on_click()
                                btn_clicked = True
                                break
                        
                        if not btn_clicked and mouse_pos[0] < SIM_WIDTH and self.state == "IDLE":
                            found_target = False
                            for i, t in enumerate(self.targets):
                                if math.sqrt((mouse_pos[0]-t[0])**2 + (mouse_pos[1]-t[1])**2) < 20:
                                    self.dragging_target_idx = i
                                    found_target = True
                                    break
                                
                            if not found_target:
                                if math.sqrt((mouse_pos[0]-self.start_pos[0])**2 + (mouse_pos[1]-self.start_pos[1])**2) < 20:
                                    self.dragging_start = True
                                else:
                                    self.drawing_obstacle_start = mouse_pos
                    elif event.button == 3: # Right click
                        if self.state == "IDLE" and mouse_pos[0] < SIM_WIDTH:
                            for obs in reversed(self.obstacles):
                                if obs.collidepoint(mouse_pos):
                                    self.obstacles.remove(obs)
                                    break
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_target_idx = -1
                        self.dragging_start = False
                        if self.drawing_obstacle_start:
                            end_pos = mouse_pos
                            x = min(self.drawing_obstacle_start[0], end_pos[0])
                            y = min(self.drawing_obstacle_start[1], end_pos[1])
                            w = abs(self.drawing_obstacle_start[0] - end_pos[0])
                            h = abs(self.drawing_obstacle_start[1] - end_pos[1])
                            if w > 10 and h > 10:
                                self.obstacles.append(pygame.Rect(x, y, w, h))
                            self.drawing_obstacle_start = None

            if self.state == "TRAINING":
                alg = self.algorithms[self.current_algo_idx]
                if alg == "NEAT" and self.population:
                    if self.current_generation < self.iterations_to_train:
                        self.population.run(self.eval_genomes, 1)
                        self.current_generation += 1
                    else:
                        self.best_genome_cache = self.population.best_genome
                        with open(get_cache_filename(self.iterations_to_train), 'wb') as f:
                            pickle.dump(self.best_genome_cache, f)
                        self.state = "IDLE"
                elif alg == "Q-LEARNING":
                    if self.current_generation < self.iterations_to_train:
                        brain = QLearningBrain.create(self.q_table, epsilon=max(0.01, 1.0 - (self.current_generation / float(max(1, self.iterations_to_train)))))
                            
                        # Generate a mini batch of testing cars
                        self.cars = [CarAI(self.start_pos[0], self.start_pos[1], None, brain) for _ in range(5)]
                        obs_tuples = [(o.x, o.y, o.width, o.height) for o in self.obstacles] if self.enable_blocks else []
                        
                        frames = 0
                        while frames < MAX_FRAMES_PER_GEN and any(c.alive for c in self.cars):
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    sys.exit()

                            for car in self.cars:
                                if car.alive:
                                    if car.target_idx < len(self.targets):
                                        t_pos = self.targets[car.target_idx]
                                        st, sp = car.think(t_pos)
                                        car.update(st, sp, obs_tuples, t_pos)
                                        if car.reached_target:
                                            car.reached_target = False
                                    else:
                                        car.alive = False

                            self.draw_sim()
                            self.draw_panel()
                            pygame.display.flip()
                            self.clock.tick(500)
                            frames += 1
                            
                        best_fitness = max((car.fitness for car in self.cars), default=0)
                        self.fitness_history.append(best_fitness)
                        self.current_generation += 1
                    else:
                        self.state = "IDLE"
            
            elif self.state == "PREVIEW" and self.cars:
                if not any(c.alive for c in self.cars):
                    self.reset_cars_for_preview()
                    
                obs_tuples = [(o.x, o.y, o.width, o.height) for o in self.obstacles] if self.enable_blocks else []
                for car in self.cars:
                    if car.alive:
                        if car.target_idx < len(self.targets):
                            t_pos = self.targets[car.target_idx]
                            st, sp = car.think(t_pos)
                            car.update(st, sp, obs_tuples, t_pos)
                            if car.reached_target:
                                car.reached_target = False
                        else:
                            car.alive = False

            self.draw_sim()
            self.draw_panel()
            pygame.display.flip()
            self.clock.tick(60)

if __name__ == "__main__":
    app = RLAppUI()
    app.run()
