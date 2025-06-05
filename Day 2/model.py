from typing import Tuple

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.experimental.continuous_space import ContinuousSpaceAgent, ContinuousSpace
import numpy as np

class AntAgent(ContinuousSpaceAgent):
    """Continuous‑space ant executing a correlated random walk.

    Parameters
    ----------
    model : AntModel
        The parent Mesa model.
    space : ContinuousSpace
        The environment in which the ant moves (torus boundary).
    position : tuple[float, float]
        Initial *(x, y)* coordinates.
    speed : float, default=1.0
        Step length per tick.
    kappa : float, default=5.0
        Concentration of the von Mises distribution controlling turning angles.
    """
    def __init__(
            self,
            model: "AntModel",
            space: ContinuousSpace,
            position: Tuple[float, float],
            speed: float = 1.0,
            kappa: float = 5.0,
    ) -> None:
        super().__init__(space, model)
        self.position = np.asarray(position, dtype=float)
        self.heading: float = self.random.uniform(-np.pi, np.pi)  # radians
        self.speed: float = float(speed)

        # Random‑Walker parameters
        self.kappa = float(kappa)

        # Foraging state
        self.carrying: bool = False

    @property
    def _ixiy(self) -> Tuple[int, int]:
        """Current grid cell index (wrap torus)."""
        x, y = self.position
        return int(x) % self.model.width, int(y) % self.model.height

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    def _move(self) -> None:
        """Translate position one step forward along the current heading."""
        dx = self.speed * np.cos(self.heading)
        dy = self.speed * np.sin(self.heading)
        self.position += (dx, dy)
        self.position = self.space.torus_correct(tuple(self.position))

    def _random_search(self) -> None:
        """Perform one correlated‑random‑walk step."""
        self._move()
        # Change heading; VM(0, kappa) –> mean 0, concentration kappa
        self.heading += self.random.vonmisesvariate(0, kappa=self.kappa)

    def _pickup_food(self) -> bool:
        ix, iy = self._ixiy
        if self.model.food[ix, iy] > 0:
            self.model.food[ix, iy] -= 1
            self.carrying = True
            return True
        return False

    def _return_to_nest(self) -> None:
        """Ballistic homing: aim directly at a nest center and move forward."""
        nest_x, nest_y = self.model._nest_center
        vec_x = nest_x - self.position[0]
        vec_y = nest_y - self.position[1]

        # Short‑circuit if we are already at nest cell
        if self.model.nest[
            int(self.position[0]) % self.model.width,
            int(self.position[1]) % self.model.height,
        ]:
            self.carrying = False
            self.heading += np.pi  # head back out next tick
            return

        # Orient towards nest
        self.heading = np.atan2(vec_y, vec_x)

        self._move()

    def step(self) -> None:
        if self.carrying:
            self._return_to_nest()
            return

        if self._pickup_food():
            # finish a step if food is found
            return

        # Perform an uninformed search as a fallback
        self._random_search()

#%%
class AntModel(Model):
    """Mesa model containing one or more `AntAgent`s in continuous space.

    Parameters
    ----------
    width : int, default=101
        Length of the *x*-dimension (environment width).
    height : int, default=101
        Length of the *y*-dimension (environment height).
    n_ants : int, default=1
        Number of `AntAgent`s to create at initialisation.
    speed : float, default=1.0
        Step length (units per tick) assigned to each ant.
    kappa : float, default=5.0
        Concentration parameter of the von Mises turning‑angle distribution.
    seed : int | None, default=None
        Random seed fed to Mesa’s RNG.  Use a fixed value for reproducible
        trajectories; `None` draws a fresh seed at every run.

    Notes
    -----
    * The space is a **torus** – ants leaving one boundary re‑enter at the
      opposite edge (periodic boundary conditions).
    * All ants start in the center ``(width/2, height/2)``; modify the
      `position` argument of `AntAgent.create_agents` if you need a different
      initial distribution.
    """
    def __init__(
            self,
            width: int = 101,
            height: int = 101,
            n_ants: int = 10,
            speed: float = 1.0,
            kappa: float = 5.0,
            seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.width, self.height = width, height

        # Continuous space
        self.space = ContinuousSpace(
            [[0, width], [0, height]],
            torus=True,
            random=self.random,
            n_agents=n_ants,
        )

        AntAgent.create_agents(
            self,
            n_ants,
            self.space,
            position=np.tile(np.array([self.width / 2, self.height / 2]), (n_ants, 1)),
            speed=speed,
            kappa=kappa,
        )

        # Grid fields
        self.food = np.zeros((width, height), dtype=int)
        self.nest = np.zeros((width, height), dtype=bool)
        self._setup_patches()

        self.datacollector = DataCollector(
            # agent_reporters={
            #     "x": lambda agent: agent.x,
            #     "y": lambda agent: agent.y,
            #     "heading": lambda agent: agent.heading,
            # },
            model_reporters={
                "remaining_food": lambda m: m.food.sum(),
            }
        )

    def _setup_patches(self) -> None:
        cx, cy = self.width / 2, self.height / 2
        self._nest_center = (cx, cy)
        for x in range(self.width):
            for y in range(self.height):
                dist = np.hypot(x - cx, y - cy)
                if dist < 5:
                    self.nest[x, y] = True

        rng, radius = self.random, 5

        def _fill_circle(center: Tuple[float, float]):
            cx, cy = center
            for x in range(self.width):
                for y in range(self.height):
                    if np.hypot(x - cx, y - cy) < radius:
                        self.food[x, y] = rng.choice([1, 2])

        _fill_circle((0.8 * self.width, self.height / 2))  # right
        _fill_circle((0.2 * self.width, 0.2 * self.height))  # lower‑left
        _fill_circle((0.1 * self.width, 0.9 * self.height))  # upper‑left


    def step(self) -> None:
        self.agents.do("step")
        self.datacollector.collect(self)