from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
try:
    from utils.EnvironmentBasics import Environment, Action, State, StateOrSid, ActionOrAid
    from utils.AgentBasics import Agent, Policy
except:
    from utils import printOnFirstCall
    from EnvironmentBasics import Environment, Action, State, StateOrSid, ActionOrAid
    from AgentBasics import Agent, Policy
from typing import *

"""
领土战争环境类，继承自Environment类
这本质上还是一个GridWorld，但是他是一个更加复杂的游戏环境
规则：
1. 游戏基于格子世界
    1.1. 格子世界中有不同类型的格子：普通格子，陷阱格子，主城格子
    1.2. 普通格子被占领后会变成占领者的领地格子
    1.3. 陷阱格子被占领后会变成普通格子
    1.4. 主城格子被占领后会变成占领者的领地格子，同时会使占领着每回合多拓展一个领地格子
    1.5. (可选)游戏进行到一定回合数后，会在地图上随机生成一个陷阱格子，这是在模拟自然灾害导致的地图变化
2. 有最多3个玩家，出生地点随机，出生点为主城格子
    2.1. 每个玩家每回合必须拓展自己的领地，即占领一个格子，占领的格子必须与自己的领地格子相邻
    2.2. 拥有n个主城格子的玩家，每回合可以拓展n个领地格子
    2.3. 玩家失去所有主城格子后，此玩家失败
    2.4. 占领整个地图的玩家获胜
3. RL算法表示
    3.1. 整个格子世界的状态为state
    3.2. 每个玩家为agent
    3.3. 每个玩家的动作为action
    3.4. 每个玩家在特定state的action为当前所有可占领的格子
    3.5. 每个玩家胜利时的reward为1，失败时的reward为-1，其他情况为0
    3.6. agent的policy不能是greedy的，因为greedy policy会使得两玩家来回占领领地格子，导致游戏无法结束
    3.7. (可选)每个玩家占领其他玩家的领地格子时，占领者的reward为0.1，被占领者的reward为-0.1，这是为了给agent一个鼓励占领其他玩家领地的动力
"""

class TerritorialWarState(State):
    EMPTY_GRID = 0
    TRAP_GRID = 1
    PLAYER_1_GRID = 2
    PLAYER_2_GRID = 3
    PLAYER_3_GRID = 4
    PLAYER_1_CAPITAL_GRID = 5
    PLAYER_2_CAPITAL_GRID = 6
    PLAYER_3_CAPITAL_GRID = 7
    def __init__(self, name:str, reward: float, representation: np.ndarray) -> None:
        """ 领土战争的State，继承自State类，是一个格子世界的状态

        :param name: 状态名
        :param rewards: 奖励，这里是一个数字，表示当前状态的奖励
        :param representation: 用于表示当前状态的numpy数组，这里是一个二维数组，每个元素表示一个格子的类型
        """
        super().__init__(name, [reward], False, representation)

        # 如果terminal state，那么整个格子世界的所有格子类型全相同
        self.is_terminal = np.all((self.representation % 3 == self.representation[0, 0] % 3))
        self.rewards = 0 if self.is_terminal else reward

    def __getitem__(self, item):
        return self.representation[item]

    def __setitem__(self, key, value):
        self.representation[key] = value
    
    def getReward(self) -> float:
        """ Get a random reward from the reward distribution

        :return: a random reward
        """
        return self.rewards

    def getNeighbors(self, row: int, col: int) -> List[int]:
        """ Get the neighbors of a grid (wrap around if on the edge

        :param row: the row of the grid
        :param col: the col of the grid
        :return: a list of neighbors (top, bottom, left, right)
        """
        H, W = self.representation.shape
        return [
            self.representation[(row - 1) % H, col],
            self.representation[(row + 1) % H, col],
            self.representation[row, (col - 1) % W],
            self.representation[row, (col + 1) % W]
        ]

    def isCapitalGrid(self, row: int, col: int) -> bool:
        return self[row, col] >= self.PLAYER_1_CAPITAL_GRID

    def isPlayerGrid(self, row: int, col: int) -> bool:
        return self[row, col] >= self.PLAYER_1_GRID

    def isTrapGrid(self, row: int, col: int) -> bool:
        return self[row, col] == self.TRAP_GRID

    def isEmptyGrid(self, row: int, col: int) -> bool:
        return self[row, col] == self.EMPTY_GRID



class TerritorialWarAction(Action):
    def __init__(self, representation: Tuple[int, int]) -> None:
        super().__init__("action", representation)

    def __repr__(self):
        return f"TerritorialWarAction({self.representation})"

    def __str__(self):
        return f"TerritorialWarAction({self.representation})"

    @staticmethod
    def fromHash(hash_val: int, world_size: Tuple[int, int]) -> TerritorialWarAction:
        H, W = world_size
        return TerritorialWarAction((hash_val // W, hash_val % W))
    
    def toHash(self, world_size: Tuple[int, int]):
        H, W = world_size
        return self.representation[0] * W + self.representation[1]


class CallFunctionLikeList():
    # 用于将函数变成可索引的类，这样就可以像访问列表一样访问函数了
    def __init__(self, func: Callable):
        self.func = func

    def __getitem__(self, item):
        return self.func(item)


class TerritorialWarEnv():
    EMPTY_GRID = 0
    TRAP_GRID = 1
    PLAYER_1_GRID = 2
    PLAYER_2_GRID = 3
    PLAYER_3_GRID = 4
    PLAYER_1_CAPITAL_GRID = 5
    PLAYER_2_CAPITAL_GRID = 6
    PLAYER_3_CAPITAL_GRID = 7
    def __init__(self, world_size: Tuple[int, int], random_seed: int) -> None:
        
        self.world_size = world_size
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # 每个格子可以是8种类型，而一共有world_size[0] * world_size[1]个格子，因此一共有8 ** (world_size[0] * world_size[1])种状态
        # 仅仅10x10的格子世界就有8**100种状态
        self.n_states = 8 ** (world_size[0] * world_size[1])
        # 如果条件合适，每个格子可以被占领，因此一共有world_size[0] * world_size[1]个动作，当然实际上只有与自己领地相邻的格子才可以被占领
        # 因此可选的action数量少于world_size[0] * world_size[1]
        self.n_actions = world_size[0] * world_size[1]
        self.n_grids = self.n_actions

        # 为了方便，我们使用Zobrist Hashing来表示状态，因为状态太多了，我们不可能把所有状态都存储在内存中
        # sid = sum(3 ** (row * world_size[1] + col) * grid_type for row in range(world_size[0]) for col in range(world_size[1]) for grid_type in range(3))

        self.start_state = TerritorialWarState("start", 0, np.random.randint(self.TRAP_GRID + 1, size=self.world_size))
        capitals = np.random.choice(self.n_actions, size=3, replace=False)
        self.start_state[capitals[0] // world_size[1], capitals[0] % world_size[1]] = self.PLAYER_1_CAPITAL_GRID
        self.start_state[capitals[1] // world_size[1], capitals[1] % world_size[1]] = self.PLAYER_2_CAPITAL_GRID
        self.start_state[capitals[2] // world_size[1], capitals[2] % world_size[1]] = self.PLAYER_3_CAPITAL_GRID
        self.current_state = self.start_state

        # transitions无法直接存储在内存中，因为一共有8 ** (world_size[0] * world_size[1])种状态
        # 如果排除规则限制的条件，则有n_states * n_actions * n_states种转移概率
        # 但是，当在一个state执行一个action时，下一个state有且只有一个，因此对于每一个state-action对，可以只存下一个state的sid
        # transition: [s0_transition, s1_transition, s2_transition, ...]
        # si_transition: [a0_transition, a1_transition, a2_transition, ...]
        # aj_transition: [s0_prob, s1_prob, s2_prob, ...]


    def reset(self) -> None:
        np.random.seed(self.random_seed)
        self.start_state = TerritorialWarState("start", 0, np.random.randint(self.TRAP_GRID + 1, size=self.world_size))
        capitals = np.random.choice(self.n_actions, size=3, replace=False)
        self.start_state[capitals[0] // self.world_size[1], capitals[0] % self.world_size[1]] = self.PLAYER_1_CAPITAL_GRID
        self.start_state[capitals[1] // self.world_size[1], capitals[1] % self.world_size[1]] = self.PLAYER_2_CAPITAL_GRID
        self.start_state[capitals[2] // self.world_size[1], capitals[2] % self.world_size[1]] = self.PLAYER_3_CAPITAL_GRID
        self.current_state = self.start_state

    def isTerminalState(self, state: TerritorialWarState) -> bool:
        return state.is_terminal

    def isValidAction(self, player: int, state: TerritorialWarState, action: TerritorialWarAction) -> bool:
        """ Check if the action is valid in the state

        :param state: the state
        :param action: the action
        :return: True if the action is valid in the state
        """
        return action in self.getValidActions(player, state)

    def getValidActions(self, player: int, state: TerritorialWarState) -> List[TerritorialWarAction]:
        """ Get the valid action ids in the state

        :param state: the state
        :return: the valid action ids
        """


        non_player_grids = np.where((state.representation != player) & (state.representation != player + 3))
        valid_actions = []
        for (i, j) in zip(*non_player_grids):
            neighbor = state.getNeighbors(i, j)
            if player in neighbor or player + 3 in neighbor:
                valid_actions.append(TerritorialWarAction((i, j)))
        return valid_actions


    @printOnFirstCall("NOTE: getModelNextState(state, action) assumes a model exists for the environment.")
    def getModelNextState(self, state: TerritorialWarState, player: int, action: TerritorialWarAction) -> TerritorialWarState:
        """ Get the next state given the current state and the action

        :param state: the current state
        :param action: the action
        :return: the next state
        """
        grid = state.representation.copy()
        if state.isTrapGrid(*action.representation):
            grid[action.representation] = self.EMPTY_GRID
        elif state.isCapitalGrid(*action.representation):
            grid[action.representation] = player + 3
        else:
            grid[action.representation] = player
        return TerritorialWarState("state", 0, grid)

    def step(self, player: int, action: TerritorialWarAction) -> Tuple[float, bool]:
        """ Take an action in the environment

        :param action: the action
        :return: the next state, the reward, and the terminal flag
        """
        grid = self.current_state.representation.copy()
        if self.current_state.isTrapGrid(*action.representation):
            grid[action.representation] = self.EMPTY_GRID
        elif self.current_state.isCapitalGrid(*action.representation):
            grid[action.representation] = player + 3
        else:
            grid[action.representation] = player
        next_state = TerritorialWarState("state", 0, grid)
        reward = next_state.getReward()
        self.current_state = next_state
        return reward, next_state.is_terminal

    def render(self):
        plt.figure(figsize=(8, 8))
        empty_grids = np.where(self.current_state.representation == self.EMPTY_GRID)
        plt.scatter(empty_grids[1], empty_grids[0], s=1000, c="#cccccc", marker="s")
        trap_grids = np.where(self.current_state.representation == self.TRAP_GRID)
        plt.scatter(trap_grids[1], trap_grids[0], s=1000, c="#000000", marker="s")
        player_1_grids = np.where((self.current_state.representation == self.PLAYER_1_CAPITAL_GRID) | (self.current_state.representation == self.PLAYER_1_GRID))
        plt.scatter(player_1_grids[1], player_1_grids[0], s=1000, c="#ff0000", marker="s")
        player_2_grids = np.where((self.current_state.representation == self.PLAYER_2_CAPITAL_GRID) | (self.current_state.representation == self.PLAYER_2_GRID))
        plt.scatter(player_2_grids[1], player_2_grids[0], s=1000, c="#00ff00", marker="s")
        player_3_grids = np.where((self.current_state.representation == self.PLAYER_3_CAPITAL_GRID) | (self.current_state.representation == self.PLAYER_3_GRID))
        plt.scatter(player_3_grids[1], player_3_grids[0], s=1000, c="#0000ff", marker="s")
        capital_grids = np.where(self.current_state.representation >= self.PLAYER_1_CAPITAL_GRID)
        plt.scatter(capital_grids[1], capital_grids[0], s=800, c="#ffffff", marker="$c$")
        plt.xlim(-0.5, self.world_size[1] - 0.5)
        plt.ylim(-0.5, self.world_size[0] - 0.5)
        plt.grid()


if __name__ == '__main__':
    tw = TerritorialWarEnv(world_size=(4, 4), random_seed=42)
    tw.render()
    plt.show()
    player = tw.PLAYER_1_GRID
    while not tw.isTerminalState(tw.current_state):
        valid_actions = tw.getValidActions(player, tw.current_state)
        take_action = np.random.choice(valid_actions)
        print(take_action)
        tw.step(tw.PLAYER_1_GRID, take_action)
        tw.render()
        plt.show()