import math
import random
from typing import Any, Protocol, Sequence, runtime_checkable

import neat  # pyright: ignore[reportMissingTypeStubs]

Point = tuple[float, float]
RectTuple = tuple[float, float, float, float]
StateVector = list[float]
QState = tuple[int, int, tuple[int, ...]]


@runtime_checkable
class BrainProtocol(Protocol):
    def activate(self, inputs: StateVector) -> tuple[float, float]:
        ...


@runtime_checkable
class QLearningProtocol(BrainProtocol, Protocol):
    def update_q(self, reward: float, new_inputs: StateVector) -> None:
        ...

# ============================================================================
# CONSTANTS
# ============================================================================
SIM_WIDTH = 900
WINDOW_HEIGHT = 750
CAR_SIZE = 20
MAX_SPEED = 4.0
TURN_SPEED = 0.15
NUM_SENSORS = 5
SENSOR_RANGE = 120
MAX_FRAMES_PER_GEN = 400

def point_in_rect(px: float, py: float, rx: float, ry: float, rw: float, rh: float) -> bool:
    return rx <= px <= rx + rw and ry <= py <= ry + rh

# ============================================================================
# AI BRAINS
# ============================================================================
class NEATBrain:
    """
    A wrapper class for a NEAT (NeuroEvolution of Augmenting Topologies) neural network.
    This brain uses biological evolution concepts to 'grow' better neural networks over time.
    """
    def __init__(self, net: Any) -> None:
        # The underlying feedforward neural network created by neat-python
        self.net: Any = net

    def activate(self, inputs: StateVector) -> tuple[float, float]:
        """Pass the environment state through the neural network to get driving commands."""
        # 1) Feed the inputs (sensors, target dist/angle) into the neural net
        outputs = self.net.activate(inputs)
        # 2) The network returns a list; we map output 0 to steering and output 1 to speed
        return outputs[0], outputs[1]

    @staticmethod
    def create(genome: Any, config: Any) -> "NEATBrain":
        """Build a NEAT brain from a genome (DNA) and NEAT config."""
        # Compile the generic 'genome' instructions into a functioning neural network
        network: Any = neat.nn.FeedForwardNetwork.create(genome, config)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        return NEATBrain(network)


class QLearningBrain:
    """
    A reinforcement learning brain using a Q-Table to remember the value of actions.
    It learns by trial and error: doing something, observing the reward, and updating its table.
    """
    def __init__(self, q_table: dict[QState, list[float]] | None = None, epsilon: float = 0.1) -> None:
        # The Q-Table maps a specific 'state' to a list of expected rewards for each possible 'action'
        self.q_table: dict[QState, list[float]] = q_table if q_table is not None else {}
        # Exploration rate: How often to try random moves instead of the best known move
        self.epsilon: float = epsilon
        # Keep track of the last decision to update its value later based on the reward received
        self.last_state: QState | None = None
        self.last_action: int | None = None
        # Learning Rate (alpha): How much new information overrides old information (0.0 to 1.0)
        self.alpha: float = 0.1
        # Discount Factor (gamma): How much the AI cares about future rewards compared to immediate rewards
        self.gamma: float = 0.9

    def discretize(self, inputs: StateVector) -> QState:
        """Convert continuous numbers (like 1.234) into discrete buckets (like 1) so they fit in a table."""
        # Bucket the angle into 8 possible sectors
        ang = int((inputs[0] + 1.0) * 4) # 0 to 8
        # Bucket the distance into 5 possible ranges
        dist = int(inputs[1] * 5)
        # Simplify sensors to just "wall nearby" (0) or "clear" (1)
        sens = tuple(0 if s < 0.5 else 1 for s in inputs[2:])
        return (ang, dist, sens)

    def activate(self, inputs: StateVector) -> tuple[float, float]:
        """Look at the current state and choose an action using the Epsilon-Greedy policy."""
        # 1) Figure out what situation (state) we are currently in
        state = self.discretize(inputs)
        
        # 2) If we've never seen this state before, create a new row in the Q-table with zero values
        if state not in self.q_table:
            self.q_table[state] = [0.0] * 9
            
        # 3) Decide whether to explore (random action) or exploit (best known action)
        if random.random() < self.epsilon:
            # Explore: pick a totally random action (0 to 8)
            action = random.randint(0, 8)
        else:
            # Exploit: find the action with the highest expected reward in this state
            action = self.q_table[state].index(max(self.q_table[state]))
            
        # 4) Remember what we decided, so we can learn from it in the next step
        self.last_state = state
        self.last_action = action
        
        # 5) Translate the "action number" (0 to 8) into actual steering and speed commands
        steering = -1 + (action % 3)
        speed = (action // 3) * 0.5
        return steering, speed
        
    def update_q(self, reward: float, new_inputs: StateVector) -> None:
        """The Bellman Equation: update the Q-Table based on the reward we just got."""
        # Don't update if we are just previewing the model (epsilon == 0) or haven't made a move yet
        if self.last_state is None or self.epsilon == 0:
            return 

        if self.last_action is None:
            return
        
        # 1) Calculate the new state we ended up in
        new_state = self.discretize(new_inputs)
        if new_state not in self.q_table:
            self.q_table[new_state] = [0.0] * 9
            
        # 2) Look forward: what's the best possible reward we can get from this NEW state?
        max_q = max(self.q_table[new_state])
        
        # 3) Update the OLD state's action value using the specific reward we just earned AND the new state's potential
        self.q_table[self.last_state][self.last_action] += self.alpha * (
            reward + self.gamma * max_q - self.q_table[self.last_state][self.last_action]
        )

    @staticmethod
    def create(q_table: dict[QState, list[float]] | None = None, epsilon: float = 0.1) -> "QLearningBrain":
        """Build a Q-learning brain with an optional shared Q-table."""
        return QLearningBrain(q_table, epsilon)

class CarAI:
    """
    A small car agent that senses walls, aims for targets, and asks a brain how to move.
    This class handles all the physics, sensors, and fitness tracking - the actual 'driving'.
    """

    def __init__(self, x: float, y: float, genome: Any = None, net: Any = None) -> None:
        # Physics / Positioning
        self.x: float = x
        self.y: float = y
        self.angle: float = 0
        self.speed: float = 0
        
        # State Tracking
        self.alive: bool = True               # Is the car still driving? (Turns false if it hits a wall)
        self.reached_target: bool = False     # Did it just hit a target on this frame?
        self.target_idx: int = 0              # Which target in the sequence is it aiming for?
        self.frames_alive: int = 0            # Timer for the current target
        
        # Sensors: Array representing distance to walls. 1.0 = clear, closer to 0.0 = getting close to a wall.
        self.sensor_readings: list[float] = [1.0] * NUM_SENSORS
        
        # Performance Tracking: Used by NEAT to decide if this car gets to reproduce
        self.fitness: float = 0
        
        # Links back to the AI making the decisions
        self.genome: Any = genome
        self.brain: BrainProtocol | None = net

    def get_corners(self) -> list[Point]:
        half_size = CAR_SIZE / 2
        cos_a: float = math.cos(self.angle)
        sin_a: float = math.sin(self.angle)
        corners: list[Point] = []
        for dx, dy in [(-half_size, -half_size), (half_size, -half_size),
                       (half_size, half_size), (-half_size, half_size)]:
            cx: float = self.x + dx * cos_a - dy * sin_a
            cy: float = self.y + dx * sin_a + dy * cos_a
            corners.append((cx, cy))
        return corners

    def update_sensors(self, obstacles: Sequence[RectTuple]) -> None:
        angles = [self.angle - math.pi/4, self.angle - math.pi/8, 
                  self.angle, 
                  self.angle + math.pi/8, self.angle + math.pi/4]
        
        for i, ray_angle in enumerate(angles):
            min_dist: float = SENSOR_RANGE
            step_size: int = 4
            for step in range(1, int(SENSOR_RANGE/step_size)):
                px: float = self.x + math.cos(ray_angle) * step * step_size
                py: float = self.y + math.sin(ray_angle) * step * step_size
                
                # Boundary check
                if px < 0 or px > SIM_WIDTH or py < 0 or py > WINDOW_HEIGHT:
                    min_dist = min(min_dist, step * step_size)
                    break
                    
                # Obstacle check
                hit: bool = False
                for obs in obstacles:
                    if point_in_rect(px, py, *obs):
                        min_dist = min(min_dist, step * step_size)
                        hit = True
                        break
                if hit: break
            
            self.sensor_readings[i] = min_dist / SENSOR_RANGE

    def check_collision(self, obstacles: Sequence[RectTuple]) -> bool:
        corners = self.get_corners()
        for cx, cy in corners:
            if cx < 0 or cx > SIM_WIDTH or cy < 0 or cy > WINDOW_HEIGHT:
                return True
            for obs in obstacles:
                if point_in_rect(cx, cy, *obs):
                    return True
        return False

    def get_state(self, target: Point) -> StateVector:
        # 1) Find the vector from the car to the current target.
        dx: float = target[0] - self.x
        dy: float = target[1] - self.y

        # 2) Turn that vector into an angle so we know where the target is.
        target_angle: float = math.atan2(dy, dx)
        angle_diff: float = target_angle - self.angle
        
        # 3) Wrap the angle difference into the range [-pi, pi].
        while angle_diff > math.pi: angle_diff -= 2 * math.pi
        while angle_diff < -math.pi: angle_diff += 2 * math.pi
        
        # 4) Measure distance to the target and scale it to a small range.
        dist_to_target: float = math.sqrt(dx*dx + dy*dy) / SIM_WIDTH

        # 5) Return the full state: target direction, target distance, sensors.
        return [angle_diff / math.pi, dist_to_target] + self.sensor_readings

    def think(self, target: Point) -> tuple[float, float]:
        """Build the current state and let the selected brain return steering and speed."""
        # 1) If the car is already crashed or finished, do nothing.
        if not self.alive:
            return 0, 0

        if self.brain is None:
            return 0, 0

        # 2) Convert the world into a compact numeric state.
        inputs: StateVector = self.get_state(target)

        # 3) Ask the brain to choose steering and speed from that state.
        steering: float
        speed: float
        steering, speed = self.brain.activate(inputs)
        return steering, speed

    def update(self, steering: float, speed: float, obstacles: Sequence[RectTuple], target: Point) -> None:
        """Move the car, refresh sensors, handle collisions, and reward progress toward the next target."""
        # 1) Skip all physics if the car is no longer active.
        if not self.alive:
            return

        # 2) Apply the brain's chosen steering and speed.
        self.angle += steering * TURN_SPEED
        self.speed = speed * MAX_SPEED

        # 3) Move the car forward using its current angle.
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.frames_alive += 1

        # 4) Refresh the distance sensors after moving.
        self.update_sensors(obstacles)
        
        # 5) Start with a tiny negative reward so wandering forever is discouraged.
        step_reward: float = -0.1
        
        # 6) If we hit a wall or obstacle, end the run and apply a penalty.
        if self.check_collision(obstacles):
            self.alive = False
            self.fitness -= 50
            step_reward = -50
            
        # 7) Measure distance to the target to see whether it was reached.
        dx: float = target[0] - self.x
        dy: float = target[1] - self.y
        if math.sqrt(dx*dx + dy*dy) < CAR_SIZE:
            self.reached_target = True
            self.target_idx += 1

            # 8) Reward fast target collection, not just eventual success.
            bonus: float = 1000 + max(0, (MAX_FRAMES_PER_GEN - self.frames_alive) * 2)
            self.fitness += bonus
            step_reward = bonus
            self.frames_alive = 0  # Reset for the next target
            
        # 9) If the brain supports Q-learning, feed the reward back into the table.
        if isinstance(self.brain, QLearningProtocol):
            new_state: StateVector = self.get_state(target)
            self.brain.update_q(step_reward, new_state)
