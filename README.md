# Educational Car AI using NEAT

This repository contains a Python-based educational simulation of cars learning to drive using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

## Project Structure

- `neat_car_ai.py`: The main entry script and Pygame graphical frontend. Handles user parameters and visualization.
- `ai_core.py`: The core AI backend, including the definitions for the car simulation and the learning brains (such as `NEATBrain`).
- `config-feedforward.txt`: Configuration file for the `neat-python` library, defining the neural network parameters, population size, mutation rates, etc.
- `app_support.py`: Supporting application logic and utilities.

## Requirements

You will need Python installed along with the following packages:
- `pygame`
- `neat-python`

You can install these dependencies using pip:
```bash
pip install pygame neat-python
```

## Usage

### Graphical Mode
To run the simulation with the Pygame UI, simply execute the main script:
```bash
python neat_car_ai.py
```

### Headless Training (CLI Mode)
If you want to train the AI without the graphical interface (e.g., on a headless server or for faster execution), you can use the `--cli` flag. You can also specify the number of iterations using `--iters`.

```bash
python neat_car_ai.py --cli --iters 50
```

## How It Works
The simulation generates a population of cars, each controlled by a neural network. As the cars attempt to navigate the track, they are evaluated based on their fitness (e.g., distance traveled without crashing). The best performing cars are selected to breed the next generation, gradually evolving highly capable driving agents.
