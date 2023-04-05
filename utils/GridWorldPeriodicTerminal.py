import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.EnvironmentBasics import Action, State, ActionOrAid
from utils.GridWorld import GridWorld
from utils.AgentBasics import Agent
from typing import *

class PeriodicRewardState(State):
    reward_i = 0
    count = 0
    interval_change_reward = 1000_000
    reward_changed = False
    def __init__(self, name: str, rewards: Union[np.ndarray, List[float]], is_terminal: bool = False, representation: Any = None) -> None:
        super().__init__(name, rewards, is_terminal, representation)
        self.n_rewards = len(rewards)

    def getReward(self) -> float:
        if self.is_terminal:
            PeriodicRewardState.updateRewardi()
        return self.rewards[self.__class__.reward_i % self.n_rewards]
    
    @classmethod
    def updateRewardi(cls):
        if cls.count % cls.interval_change_reward == 0:
            cls.reward_i += 1
            cls.reward_changed = True
        cls.count += 1


class GridWorldPeriodicTerminal(GridWorld):
    def __init__(self, world_size: int, win_states: List[Tuple[int, int]] = None, fail_states: List[Tuple[int, int]] = None, period: int = 1000_000):
        """ Rewrite GridWorld class, make state rewards more random

        :param world_size: size of the world
        """
        PeriodicRewardState.interval_change_reward = period
        state_space = []
        n_win_states = len(win_states)
        i = 0
        for row in range(world_size):
            for col in range(world_size):
                if (row, col) in win_states:
                    is_terminal = True
                    rewards = [0.2] * n_win_states
                    rewards[i] = 1
                    i += 1
                    representation = "W"
                elif (row, col) in fail_states:
                    is_terminal = True
                    rewards = [-2]
                    representation = "F"
                else:
                    is_terminal = False
                    rewards = [0]
                    representation = " "
                state_space.append(PeriodicRewardState(f"({row},{col})", rewards, is_terminal, representation=representation))

        action_space = [Action("left"), Action("right"), Action("down"), Action("up")]
        super().__init__(world_size, win_states + fail_states, state_space, action_space)
        self.path = [self.current_sid]

    def render(self, agent:Agent=None, figsize:Tuple[int, int]=None):
        """ Render the environment, override method in Environment class
        1. draw the states as grids
        2. no need to draw transitions, because they are obvious in the grid world
        3. highlight current state
        4. highlight start state
        5. highlight terminal states
        """
        # if figsize is None:
        #     figsize = (5, 5)
        # plt.figure(figsize=figsize)

        fig = Figure(figsize=figsize)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        plt.xlim(-1, self.world_size)
        plt.ylim(-1, self.world_size)
        color = agent.state_value if agent is not None else [state.is_terminal for state in self.states]
        ax.scatter(self.grid_colid, self.grid_rowid, c=color, marker="s", s=1000, alpha=0.75, cmap="winter", vmin=0, vmax=1)
        path = [self.stateIdToCoord(sid) for sid in self.path]
        ax.plot([p[1] for p in path], [p[0] for p in path], c="#000000", linewidth=2)
        for gi in range(self.n_grids):
            label = f"{agent.state_value[gi] if agent is not None else self.states[gi].is_terminal:.2f}"
            ax.annotate(label, (self.grid_colid[gi], self.grid_rowid[gi]), textcoords="offset points", xytext=(2, -2), ha="center")

        self.highlightTerminalStates(ax)
        if agent is not None:
            for sid in range(self.n_states):
                if not self.isTerminalState(sid):
                    for action in self.getValidActionIds(sid):
                        row, col = self.stateIdToCoord(sid)
                        self.drawArrow(ax, col, row, action, agent.policy[sid, action])

        canvas.draw()       # draw the canvas, cache the renderer
        return np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))

    def step(self, action: ActionOrAid) -> float:
        """
        Step to the next state according to the current state and the action taken
        :param action: action id or action
        :return: next state id, reward, is_terminal
        """
        assert self.isValidAction(self.current_sid, action), f"Invalid action {action} in state {self.current_sid}"
        # According to the current state and the action taken, choose one of the next states based on the probabilities
        next_sid = np.random.choice(self.n_states, p=self.transitions[self.current_sid, action, :])
        next_state = self.sid_to_state[next_sid]
        reward = next_state.getReward()
        self.current_sid = next_sid
        self.path.append(next_sid)
        return reward

    def highlightTerminalStates(self, ax):
        """ Highlight terminal states """
        for sid in self.getTerminalStateIds():
            row, col = self.stateIdToCoord(sid)
            if self.sid_to_state[self.coordToStateId(row, col)].representation == "W":
                ax.scatter(col, row, c="green", marker="s", s=700)
            elif self.sid_to_state[self.coordToStateId(row, col)].representation == "F":
                ax.scatter(col, row, c="red", marker="s", s=700)

    
    def drawArrow(self, ax, x: int, y: int, action: Union[Action, int], length: float = 1):
        """ Draw an arrow from point x, y, direction code: {0: "left", 1: "right", 2: "down", 3: "up"} """
        if not isinstance(action, Action):
            action = self.aid_to_action[action]
        dx, dy = 0, 0
        if action.name == "left":
            dx = np.clip(-1 * length, 0, 0.9)
        elif action.name == "right":
            dx = np.clip(1 * length, 0, 0.9)
        elif action.name == "down":
            dy = np.clip(-1 * length, 0, 0.9)
        elif action.name == "up":
            dy = np.clip(1 * length, 0, 0.9)

        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, length_includes_head=True)