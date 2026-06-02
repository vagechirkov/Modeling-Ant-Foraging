from typing import Tuple
from mesa.experimental.continuous_space import ContinuousSpaceAgent, ContinuousSpace
from mesa import Model
from mesa.datacollection import DataCollector
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

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1])

    def _move(self) -> None:
        """Translate position one step forward along current heading."""
        dx = self.speed * np.cos(self.heading)
        dy = self.speed * np.sin(self.heading)
        self.position += (dx, dy)
        self.position = self.space.torus_correct(tuple(self.position))

    def _random_search(self) -> None:
        """Perform one correlated‑random‑walk step."""
        self._move()
        # Change heading; VM(0, kappa) –> mean 0, concentration kappa
        self.heading += self.random.vonmisesvariate(0, kappa=self.kappa)

    def step(self) -> None:
        self._random_search()


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
            n_ants: int = 1,
            speed: float = 1.0,
            kappa: float = 5.0,
            rng: int | None = None,
    ) -> None:
        super().__init__(rng=rng)
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

        def compute_polarization(model):
            # polarization measures how synchronized agents' headings are
            # 0 (completely random) to 1 (all moving in the exact same direction)
            if len(model.agents) == 0:
                return 0.0
            cos_sum = sum(np.cos(a.heading) for a in model.agents)
            sin_sum = sum(np.sin(a.heading) for a in model.agents)
            return np.sqrt(cos_sum**2 + sin_sum**2) / len(model.agents)

        self.datacollector = DataCollector(
            agent_reporters={
                "x": lambda agent: agent.x,
                "y": lambda agent: agent.y,
                "heading": lambda agent: agent.heading,
            },
            model_reporters={
                "polarization": compute_polarization,
            }
        )
        self.running = True
        self.datacollector.collect(self)

    def step(self) -> None:
        self.agents.do("step")
        self.datacollector.collect(self)


if __name__ == "__main__":
    N_STEPS = 200  # simulation length
    RANDOM_SEED = 42
    model = AntModel(width=101, height=101, speed=1.0, kappa=5.0, rng=RANDOM_SEED)
    for _ in range(N_STEPS):
        model.step()
