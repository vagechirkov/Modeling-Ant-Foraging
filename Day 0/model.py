from mesa import Model
from mesa.discrete_space import CellAgent, OrthogonalVonNeumannGrid


class TreeCell(CellAgent):
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell
        self.condition = "Fine"

    def step(self):
        if self.condition == "On Fire":
            for agent in self.cell.get_neighborhood(radius=2).agents:
                if isinstance(agent, TreeCell) and agent.condition == "Fine":
                    agent.condition = "On Fire"
            
            self.condition = "Burned Out"


class ForestFire(Model):
    def __init__(self, width=50, height=50, density=0.6):
        super().__init__()

        self.grid = OrthogonalVonNeumannGrid(
            (width, height), 
            torus=False, 
            random=self.random
        )

        for cell in self.grid.all_cells:
            if self.random.random() < density:
                tree = TreeCell(self, cell)
                if cell.coordinate[0] == 0:
                    tree.condition = "On Fire"

    def step(self):
        self.agents.shuffle_do("step")


if __name__ == "__main__":
    model = ForestFire(width=50, height=50, density=0.6)
    # Run for 20 steps
    for i in range(20):
        model.step()
        
        # We can use AgentSet's built-in selection filters now too!
        burning_trees = len(model.agents.select(lambda t: t.condition == "On Fire"))
        print(f"Step {i}: {burning_trees} trees on fire")
