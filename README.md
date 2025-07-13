Traffic Light Optimization at a Real-World Intersection Using Reinforcement Learning

This group project focuses on optimizing the operation of traffic lights at a real intersection in Wrocław, Poland. The simulation was built using the **SUMO** (Simulation of Urban MObility) platform and enhanced with **reinforcement learning** techniques.

## Project Overview

- **Goal**: Minimize the total vehicle waiting time at the intersection during simulation.
- **Approach**: Reinforcement learning with **Deep Q-Networks (DQN)**.
- **Simulation Platform**: SUMO.
- **Interface**: Developed in **Tkinter**, allowing users to run simulations and adjust neural network parameters.

## Real-World Data Integration

- Traffic data was collected using the **TomTom API**, reflecting actual vehicle flow at different hours of the day.
- The simulated intersection corresponds to a real junction in **Wrocław**, Poland.
- **Traffic light phase durations** were measured manually on-site to accurately replicate baseline signal timings.

## Key Features

- DQN-based agent for adaptive traffic signal control.
- Custom GUI (Tkinter) for interacting with the simulation.
- Ability to configure neural network parameters (e.g., learning rate, hidden layers).
- Realistic traffic patterns based on external data sources.

## Technologies Used

- Python
- SUMO
- Tkinter
- NumPy, Pandas
- TomTom API


