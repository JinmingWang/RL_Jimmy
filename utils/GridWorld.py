import matplotlib.pyplot as plt
import numpy as np
try:
    from utils.EnvironmentBasics import Environment, Action, State, StateOrSid, ActionOrAid
    from utils.AgentBasics import Agent, Policy
except:
    from EnvironmentBasics import Environment, Action, State, StateOrSid, ActionOrAid
    from AgentBasics import Agent, Policy
from typing import *

class GridWorld(Environment):
    def __init__(self, world_size: int, 
                 terminal_states: List[Tuple[int, int]] = None, 
                 state_space: List[State] = None, 
                 action_space: List[Action] = None):
        """ Grid world class

        :param world_size: size of the world
        """
        if state_space is None:
            state_space = []
            for row in range(world_size):
                for col in range(world_size):
                    is_terminal = (row, col) in terminal_states
                    state_space.append(State(f"({row},{col})", [int(is_terminal)], is_terminal))
        if action_space is None:
            action_space = [Action("left"), Action("right"), Action("down"), Action("up")]
        super().__init__(state_space, action_space)

        self.world_size = world_size
        self.n_grids = world_size ** 2
        self.grid_rowid = np.arange(self.n_grids)//self.world_size
        self.grid_colid = np.arange(self.n_grids)%self.world_size

        # Define transitions of the grid world
        state_id = 0
        for row in range(world_size):
            for col in range(world_size):
                self.setStateTransition(state_id, 0, self.coordToStateId(row, (col-1) % world_size), 1)  # left
                self.setStateTransition(state_id, 1, self.coordToStateId(row, (col+1) % world_size), 1)  # right
                self.setStateTransition(state_id, 2, self.coordToStateId((row-1) % world_size, col), 1)  # down
                self.setStateTransition(state_id, 3, self.coordToStateId((row+1) % world_size, col), 1)  # up
                state_id += 1

        self.path = [self.start_sid]


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
    

    def reset(self, start_state_id: int=-1) -> int:
        """ Reset the environment to the start state
        If start_state_id is not specified or set to -1, choose a random start state

        :param start_state_id: the id of the start state, defaults to -1
        :return: the start state id
        """
        if start_state_id == -1:
            self.start_sid = np.random.randint(self.n_states)
        else:
            self.start_sid = start_state_id
        self.current_sid = self.start_sid
        self.path = [self.current_sid]
        return self.start_sid
    

    def getReward(self, state_id: int) -> float:
        """ Get the reward of the state """
        return self.sid_to_state[state_id].getReward()


    def coordToStateId(self, row: int, col: int) -> int:
        return row * self.world_size + col
    

    def stateIdToCoord(self, state_id: int) -> Tuple[int, int]:
        return state_id // self.world_size, state_id % self.world_size
    

    def isTerminalCoord(self, row: int, col: int) -> bool:
        return self.sid_to_state[self.coordToStateId(row, col)].is_terminal
    

    def getValidActionIds(self, state: StateOrSid) -> List[int]:
        return [0, 1, 2, 3]


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
        for gi in range(self.n_grids):
            label = f"{agent.state_value[gi] if agent is not None else self.states[gi].is_terminal:.2f}"
            plt.annotate(label, (self.grid_colid[gi], self.grid_rowid[gi]), textcoords="offset points", xytext=(2, -2), ha="center")

        self.highlightTerminalStates()
        if agent is not None:
            for sid in range(self.n_states):
                if not self.isTerminalState(sid):
                    for action in agent.getBestActions(sid):
                        row, col = self.stateIdToCoord(sid)
                        self.drawArrow(col, row, action)

    @property
    def shape(self):
        # returns the shape of the grid
        return (self.world_size, self.world_size)
        

    def drawArrow(self, x: int, y: int, action: Union[Action, int], length: float = 1):
        """ Draw an arrow from point x, y, direction code: {0: "left", 1: "right", 2: "down", 3: "up"} """
        if not isinstance(action, Action):
            action = self.aid_to_action[action]
        dx, dy = 0, 0
        if action.name == "left":
            dx = -0.4 * length
        elif action.name == "right":
            dx = 0.4 * length
        elif action.name == "down":
            dy = -0.4 * length
        elif action.name == "up":
            dy = 0.4 * length

        plt.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, length_includes_head=True)


    def highlightTerminalStates(self):
        """ Highlight terminal states """
        for sid in self.getTerminalStateIds():
            row, col = self.stateIdToCoord(sid)
            plt.scatter(col, row, c="red", marker="s", s=700)


if __name__ == "__main__":
    grid = GridWorld(4, terminal_states=[(0, 0), (3, 3)])
    print(grid.getValidActionIds(1))
    grid.render()
    plt.show()

    policy = Policy(grid)