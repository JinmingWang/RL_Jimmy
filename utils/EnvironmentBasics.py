from __future__ import annotations
import numpy as np
from typing import *
import random
import matplotlib.pyplot as plt
import networkx as nx


"""
one state - multiple actions
one action - multiple next states
one next state - multiple rewards

Have to store each state-action pair as a separate object
"""


class Action():
    def __init__(self, name:str, representation:Any=None) -> None:
        """ Action class

        :param name: name of the action
        """
        self.name = name
        self.representation = representation

    def __repr__(self) -> str:
        return f"Action({self.name})"

    def __str__(self) -> str:
        return self.__repr__()
    
    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, value: Union[Action, str]) -> bool:
        # name equality
        if isinstance(value, Action):
            return self.name == value.name
        elif isinstance(value, str):
            return self.name == value


class State():
    def __init__(self, name:str, rewards: Union[np.ndarray, List[float]], is_terminal: bool=False) -> None:
        """ State class

        :param name: name of the state
        :param rewards: rewards of this state
        :param is_terminal: is this a terminal state, defaults to False
        """
        self.name = name
        self.is_terminal = is_terminal
        self.rewards = rewards

    def __repr__(self) -> str:
        return f"State({self.name} with rewards: {self.rewards}, is_terminal: {self.is_terminal})"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def getReward(self) -> float:
        """ Get a random reward from the reward distribution

        :return: a random reward
        """
        return np.random.choice(self.rewards)


ActionOrAid = Union[Action, int]
StateOrSid = Union[State, int]
StateAction = Tuple[State, Action]

class Environment():
    def __init__(self, state_space: List[State], action_space: List[Action], start_state_id: int=-1) -> None:
        self.n_states = len(state_space)
        self.n_actions = len(action_space)
        self.sid_to_state: List[State] = state_space
        self.state_to_sid: Dict[State, int] = {state: i for i, state in enumerate(state_space)}
        self.aid_to_action: List[Action] = action_space
        self.action_to_aid: Dict[Action, int] = {action: i for i, action in enumerate(action_space)}

        self.start_sid = start_state_id if start_state_id != -1 else np.random.randint(self.n_states)
        self.current_sid = self.start_sid

        # transition: [s0_transition, s1_transition, s2_transition, ...]
        # si_transition: [a0_transition, a1_transition, a2_transition, ...]
        # aj_transition: [s0_prob, s1_prob, s2_prob, ...]
        self.transitions: np.ndarray = np.zeros((len(state_space), len(action_space), len(state_space)))


    @property
    def states(self) -> List[State]:
        return self.sid_to_state
    
    @property
    def state_ids(self) -> np.ndarray:
        return np.arange(self.n_states, dtype=np.int32)

    @property
    def actions(self) -> List[Action]:
        return self.aid_to_action
    
    @property
    def action_ids(self) -> np.ndarray:
        return np.arange(self.n_actions, dtype=np.int32)

    @property
    def state_actions(self) -> List[Tuple[State, Action]]:
        return [(state, action) for state in self.states for action in self.actions]

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
        return self.start_sid


    def __getitem__(self, indices: Tuple[StateOrSid, ActionOrAid, StateOrSid]) -> float:
        """ Get the transition probability of the state-action-state triple

        :param indices: (state_id, action_id, next_state_id)
        :return: the transition probability
        """
        if isinstance(indices[0], State):
            indices = (self.state_to_sid[indices[0]], indices[1], indices[2])
        if isinstance(indices[1], Action):
            indices = (indices[0], self.action_to_aid[indices[1]], indices[2])
        if isinstance(indices[2], State):
            indices = (indices[0], indices[1], self.state_to_sid[indices[2]])
        return self.transitions[indices]


    def __setitem__(self, indices: Tuple[StateOrSid, ActionOrAid, StateOrSid], value: float) -> None:
        """ Set the transition probability of the state-action-state triple

        :param indices: (state_id, action_id, next_state_id)
        :param value: the transition probability
        """
        if isinstance(indices[0], State):
            indices = (self.state_to_sid[indices[0]], indices[1], indices[2])
        if isinstance(indices[1], Action):
            indices = (indices[0], self.action_to_aid[indices[1]], indices[2])
        if isinstance(indices[2], State):
            indices = (indices[0], indices[1], self.state_to_sid[indices[2]])
        self.transitions[indices] = value


    def setStateTransition(self, state: StateOrSid, action: ActionOrAid, next_state: StateOrSid, prob: float) -> None:
        """ Set the transition probability of the state-action-state triple

        :param state: the state
        :param action: the action
        :param next_state: the next state
        :param prob: the transition probability
        """
        if isinstance(state, State):
            state = self.state_to_sid[state]
        if isinstance(action, Action):
            action = self.action_to_aid[action]
        if isinstance(next_state, State):
            next_state = self.state_to_sid[next_state]
        self.transitions[state, action, next_state] = prob


    def setStateTransitions(self, state: StateOrSid, action: ActionOrAid, next_states: List[StateOrSid], probs: List[float]) -> None:
        """ Set the transition probabilities of the state-action-state triples

        :param state: the state
        :param action: the action
        :param next_states: the next states
        :param probs: the transition probabilities
        """
        if isinstance(state, State):
            state = self.state_to_sid[state]
        if isinstance(action, Action):
            action = self.action_to_aid[action]
        if isinstance(next_states[0], State):
            next_states = [self.state_to_sid[state] for state in next_states]
        self.transitions[state, action, next_states] = probs

    
    def isTerminalState(self, state: StateOrSid) -> bool:
        """ Check if the state is a terminal state

        :param state: the state
        :return: True if the state is a terminal state
        """
        if isinstance(state, int):
            state = self.sid_to_state[state]
        return state.is_terminal


    def isValidAction(self, state: StateOrSid, action: ActionOrAid) -> bool:
        """ Check if the action is valid in the state

        :param state: the state
        :param action: the action
        :return: True if the action is valid in the state
        """
        if isinstance(state, State):
            state = self.state_to_sid[state]
        if isinstance(action, Action):
            action = self.action_to_aid[action]
        return self.transitions[state, action, :].sum() != 0


    def getValidActionIds(self, state: StateOrSid) -> np.ndarray:
        """ Get the valid action ids in the state

        :param state: the state
        :return: the valid action ids
        """
        if isinstance(state, State):
            state = self.state_to_sid[state]
        return np.nonzero(self.transitions[state, :, :].sum(axis=1))[0]
    

    def getNextStateProbs(self, state: StateOrSid, action: ActionOrAid) -> np.ndarray:
        """ Get the net state probabilities of the state-action-state triples

        :param state: the state
        :param action: the action
        :return: the net state probabilities
        """
        if isinstance(state, State):
            state = self.state_to_sid[state]
        if isinstance(action, Action):
            action = self.action_to_aid[action]
        return self.transitions[state, action, :] / self.transitions[state, action, :].sum()
    

    def getTerminalStateIds(self) -> np.ndarray:
        """ Get the terminal state ids

        :return: the terminal state ids
        """
        return np.array([sid for sid in self.state_ids if self.states[sid].is_terminal])


    def step(self, action: ActionOrAid) -> Tuple[Union[int, State], float, bool]:
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
        return next_sid, reward, next_state.is_terminal
    
    def render(self):
        """ Render the environment
        1. draw the states and actions in different colors as nodes
        2. draw the transitions with probability as edges
        3. highlight current state
        4. highlight start state
        5. highlight terminal states
        """
        plt.figure(figsize=(10, 5))
        G = nx.DiGraph()
        pos = dict()
        # Draw all states
        for state in self.states:
            # Add the state as a node
            G.add_node(state.name, color="cyan", bipartite=0)
            # Add the state to the left side of the graph
            pos[state.name] = (self.state_to_sid[state], 0)
            if state.is_terminal:
                plt.scatter(*pos[state.name], color="red", s=500, marker="s")
            if self.state_to_sid[state] == self.start_sid:
                plt.scatter(*pos[state.name], color="green", s=500, marker="D")
            if self.state_to_sid[state] == self.current_sid:
                plt.scatter(*pos[state.name], color="blue", s=600, marker="8")

        # Iterate over all actions of all the states
        action_offset = 0.5
        for (state, action) in self.state_actions:
            # Add the action as a node
            G.add_node(action.name, color="orange", bipartite=1)
            # Add the action to the right side of the graph
            pos[action.name] = (self.action_to_aid[action] + action_offset, 1)
            # Iterate over all the next states of the action
            for next_sid in range(self.n_states):
                if self[state, action, next_sid] != 0:
                    # Add the edge from the state to the action
                    G.add_edge(state.name, action.name, color="cyan", weight=1)
                    # Add the edge from the action to the next state
                    G.add_edge(action.name, self.sid_to_state[next_sid].name, color="orange", weight=self[state, action, next_sid])

        nx.draw_networkx_nodes(G, pos, node_color=[attr["color"] for node, attr in G.nodes.items()])
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color=[edge[2]["color"] for edge in G.edges(data=True)], width=[edge[2]["weight"]*5 for edge in G.edges(data=True)])
        # Also display edge weights for action to state transitions
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True) if G.nodes[v]["bipartite"] == 0}
        # edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        bbox = dict(facecolor='orange', edgecolor='none', alpha=1, boxstyle='round,pad=0.2')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.75, bbox=bbox, font_color="white", font_size=7)

        nx.draw_networkx_labels(G, pos)

        plt.xlim(-0.2, self.n_actions - 0.3)
        plt.ylim(-0.2, 1.2)

        

if __name__ == "__main__":
    state_space = [State("s1", [1, 2, 3]), State("s2", [4, 5, 6]), State("s3", [7, 8, 9]),
                   State("s4", [10, 11, 12], is_terminal=True)]
    action_space = [Action("a1"), Action("a2"), Action("a3"), Action("a4")]
    test_env = Environment(state_space, action_space)

    test_env.setStateTransitions(test_env.states[0], action_space[0], [test_env.states[1], test_env.states[2]], [0.7, 0.3])
    test_env.setStateTransitions(test_env.states[1], action_space[1], [test_env.states[0], test_env.states[2]], [0.51, 0.49])
    test_env.setStateTransitions(test_env.states[2], action_space[2], [test_env.states[0], test_env.states[1]], [0.2, 0.8])
    test_env.setStateTransitions(test_env.states[2], action_space[3], [test_env.states[3]], [1])

    print(test_env.getValidActionIds(test_env.states[1]))
    test_env.render()
    plt.show()
