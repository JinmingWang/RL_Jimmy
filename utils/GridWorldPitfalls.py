import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.EnvironmentBasics import Action, State, ActionOrAid
from utils.GridWorld import GridWorld
from utils.AgentBasics import Agent
from typing import *

# Grid World with Pitfalls
# 有些terminal state是坑，如果走到坑里就会死掉，episode结束，得到的reward是-1。

class GridWorldWithPitfalls(GridWorld):
    def __init__(self, world_size: int, win_states: List[Tuple[int, int]] = None, fail_states: List[Tuple[int, int]] = None):
        """ Rewrite GridWorld class, make state rewards more random

        :param world_size: size of the world
        """
        state_space = []
        for row in range(world_size):
            for col in range(world_size):
                if (row, col) in win_states:
                    is_terminal = True
                    rewards = [1]
                    representation = "W"
                elif (row, col) in fail_states:
                    is_terminal = True
                    rewards = [-1]
                    representation = "F"
                else:
                    is_terminal = False
                    rewards = [0]
                    representation = " "
                state_space.append(State(f"({row},{col})", rewards, is_terminal, representation=representation))

        action_space = [Action("left"), Action("right"), Action("down"), Action("up")]
        super().__init__(world_size, win_states + fail_states, state_space, action_space)
        self.path = [self.start_sid]

    def render(self, agent:Agent=None, figsize:Tuple[int, int]=None):
        """ Render the environment, override method in Environment class
        1. draw the states as grids
        2. no need to draw transitions, because they are obvious in the grid world
        3. highlight current state
        4. highlight start state
        5. highlight terminal states
        """
        if figsize is None:
            figsize = (5, 5)
        plt.figure(figsize=figsize)

        plt.xlim(-1, self.world_size)
        plt.ylim(-1, self.world_size)
        color = agent.state_value if agent is not None else [state.is_terminal for state in self.states]
        plt.scatter(self.grid_colid, self.grid_rowid, c=color, marker="s", s=1000, alpha=0.75)
        
        path = [self.stateIdToCoord(sid) for sid in self.path]
        print(111, path)
        plt.plot([p[1] for p in path], [p[0] for p in path], c="#000000", linewidth=2)
        # for gi in range(self.n_grids):
        #     label = f"{agent.state_value[gi] if agent is not None else self.states[gi].is_terminal:.2f}"
        #     plt.annotate(label, (self.grid_colid[gi], self.grid_rowid[gi]), textcoords="offset points", xytext=(2, -2), ha="center")

        self.highlightTerminalStates()
        if agent is not None:
            for sid in range(self.n_states):
                if not self.isTerminalState(sid):
                    for action in self.getValidActionIds(sid):
                        row, col = self.stateIdToCoord(sid)
                        self.drawArrow(col, row, action, agent.policy[sid, action])

    def highlightTerminalStates(self):
        """ Highlight terminal states """
        for sid in self.getTerminalStateIds():
            row, col = self.stateIdToCoord(sid)
            if self.sid_to_state[self.coordToStateId(row, col)].representation == "W":
                plt.scatter(col, row, c="green", marker="s", s=700)
            elif self.sid_to_state[self.coordToStateId(row, col)].representation == "F":
                plt.scatter(col, row, c="red", marker="s", s=700)