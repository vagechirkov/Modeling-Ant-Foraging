# Concepts & Glossary

This page maps the theoretical biological and mathematical concepts discussed in the seminar directly to their implementations in our Python codebase using the Mesa framework.

## Agent-Based Modeling (Mesa)
Agent-Based Models (ABMs) simulate the actions and interactions of autonomous agents to assess their effects on the system as a whole.
- **Agent:** Represented by the `AntAgent` class. This is the individual ant that moves, senses, and interacts.
- **Model:** Represented by the `AntModel` class. It holds the core schedule (the clock), manages the space, and coordinates all the agents.
- **Space:** We primarily use `ContinuousSpace` to allow ants to move in any direction with floating-point coordinates, rather than being locked to a rigid square grid.

## Correlated Random Walk (CRW)
A random walk where the direction of the next step is correlated with the direction of the previous step, resulting in forward persistence rather than purely chaotic movement.
- **Implementation:** Ants maintain a `self.heading`. At each step, they draw a random turning angle to update their heading.
- **von Mises Distribution:** The mathematical distribution we use to calculate the turning angle. The `kappa` ($\kappa$) parameter dictates how concentrated the turns are around 0. A high `kappa` means straight lines; a low `kappa` approaches a pure random walk.

## Central Place Foraging
A foraging strategy where an animal ventures out from a central location (the nest) to find food, and then returns to that exact same central location once it is carrying food.
- **Implementation:** The `carrying` boolean flag (True/False) tracks if the ant has found food. When `carrying == True`, the ant's movement logic switches from exploring to homing directly back to the nest coordinates.

## Stigmergy & Trail Formation
Stigmergy is a mechanism of indirect coordination through the environment. Ants leave traces (pheromones) that stimulate subsequent actions by other ants.
- **Pheromone Grid:** Implemented as a 2D NumPy array overlaying the continuous space. When ants move, they drop pheromone in the grid cell corresponding to their current position.
- **Diffusion & Evaporation:** At every tick of the model, the pheromone values in the grid slowly decay (evaporate) and spread to neighboring cells (diffuse), allowing trails to dissipate if not actively reinforced.
