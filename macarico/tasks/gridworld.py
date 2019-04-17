from dataclasses import dataclass
import random
from typing import Dict, Set

from torch.autograd import Variable
import torch.nn as nn


from macarico import DynamicFeatures, Env, Example, Loss, Policy
import macarico.util as util
from macarico.util import Varng


@dataclass(frozen=True)
class GridPosition:
    x: int
    y: int


@dataclass
class GridSettings:
    width: int
    height: int
    start: GridPosition
    walls: Set[GridPosition]
    terminals: Dict[GridPosition, int]
    per_step_cost: float
    max_steps: int
    gamma: float
    p_step_success: float


def make_default_gridworld(per_step_cost: float = 0.05,
                           max_steps: int = 50,
                           gamma: float = 0.99,
                           p_step_success: float = 0.8,
                           start_random: bool = False):
    """
    #    0123
    #   0   +
    #   1 # -
    #   2 #
    #   3
    """
    start = GridPosition(0, 3)
    if start_random:
        start = GridPosition(random.randint(0, 3), random.randint(0, 3))
    walls = {GridPosition(1, 1), GridPosition(1, 2)}
    terminals = {GridPosition(3, 0): 1, GridPosition(3, 1): -1}
    return GridWorld(GridSettings(4, 4, start, walls, terminals, per_step_cost,
                                  max_steps, gamma, p_step_success))


def make_big_gridworld(per_step_cost: float = 0.01,
                       max_steps: int = 200,
                       gamma: float = 0.99,
                       p_step_success: float = 0.9):
    """ from http://cs.stanford.edu/people/karpathy/reinforcejs/ """
    start = GridPosition(0, 9)
    walls = {GridPosition(1, 2), GridPosition(2, 2), GridPosition(3, 2),
             GridPosition(4, 2), GridPosition(6, 2), GridPosition(7, 2),
             GridPosition(8, 2), GridPosition(4, 3), GridPosition(4, 4),
             GridPosition(4, 5), GridPosition(4, 6), GridPosition(4, 7)}
    terminals = {GridPosition(3, 3): -1, GridPosition(3, 7): -1,
                 GridPosition(5, 4): -1, GridPosition(5, 5): 1,
                 GridPosition(6, 5): -1, GridPosition(6, 6): -1,
                 GridPosition(5, 7): -1, GridPosition(6, 7): -1,
                 GridPosition(8, 5): -1, GridPosition(8, 6): -1}
    return GridWorld(GridSettings(10, 10, start, walls, terminals,
                                  per_step_cost, max_steps, gamma,
                                  p_step_success))


class GridWorld(Env):
    UP = 0
    LEFT = 1
    DOWN = 2
    RIGHT = 3

    def __init__(self, settings: GridSettings):
        self.settings = settings
        self.loc = self.settings.start
        self.discount = 1
        self.actions = {GridWorld.UP, GridWorld.DOWN, GridWorld.LEFT,
                        GridWorld.RIGHT}
        super().__init__(len(self.actions), self.settings.max_steps)
        self.example.reward = 0

    def _rewind(self):
        self.loc = self.settings.start
        self.discount = 1
        self.example.reward = 0
        
    def _run_episode(self, policy: Policy) -> str:
        for _ in range(self.horizon()):
            a = policy(self)
            self.step(a)
            self.example.reward -= self.discount * self.settings.per_step_cost
            if self.loc in self.settings.terminals:
                self.example.reward += self.discount * \
                                       self.settings.terminals[self.loc]
                break
            self.discount *= self.settings.gamma
        return self.output()

    def output(self) -> str:
        return ''.join(map(self.str_direction, self._trajectory))

    def str_direction(self, a) -> str:
        return {
            GridWorld.UP: 'U',
            GridWorld.DOWN: 'D',
            GridWorld.LEFT: 'L',
            GridWorld.RIGHT: 'R'
        }.get(a, '?')
        
    def step(self, a: int) -> None:
        if random.random() > self.settings.p_step_success:
            # step failure; pick a neighboring action
            a = (a + 2 * ((random.random() < 0.5) - 1)) % 4
        # take the step
        move = {
            GridWorld.UP: [0, -1],
            GridWorld.DOWN: [0, 1],
            GridWorld.LEFT: [-1, 0],
            GridWorld.RIGHT: [1, 0]
        }.get(a, [0, 0])
        new_loc = GridPosition(self.loc.x + move[0], self.loc.y + move[1])
        if self.is_legal(new_loc):
            self.loc = new_loc
            
    def is_legal(self, new_loc: GridPosition) -> bool:
        return 0 <= new_loc.x < self.settings.width and \
               0 <= new_loc.y < self.settings.height and \
               new_loc not in self.settings.walls


class GridLoss(Loss):
    def __init__(self):
        super().__init__('reward')

    def evaluate(self, example: Example):
        return -example.reward


class GlobalGridFeatures(DynamicFeatures):
    def __init__(self, width: int, height: int):
        super().__init__(width*height)
        self.width = width
        self.height = height
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state: GridWorld) -> Variable:
        view = util.zeros(self._t.weight, 1, 1, self.dim)
        view[0, 0, state.loc.x * state.settings.height + state.loc.y] = 1
        return Varng(view)

    def __call__(self, state: GridWorld):
        return self.forward(state)


class LocalGridFeatures(DynamicFeatures):
    def __init__(self):
        super().__init__(4)
        self._t = nn.Linear(1, 1, bias=False)

    def _forward(self, state: GridWorld) -> Variable:
        view = util.zeros(self._t.weight, 1, 1, self.dim)
        if not state.is_legal(GridPosition(state.loc.x-1, state.loc.y)):
            view[0, 0, 0] = 1
        if not state.is_legal(GridPosition(state.loc.x+1, state.loc.y)):
            view[0, 0, 1] = 1
        if not state.is_legal(GridPosition(state.loc.x, state.loc.y-1)):
            view[0, 0, 2] = 1
        if not state.is_legal(GridPosition(state.loc.x, state.loc.y+1)):
            view[0, 0, 3] = 1
        return Varng(view)
    
    def __call__(self, state: GridWorld):
        return self.forward(state)
