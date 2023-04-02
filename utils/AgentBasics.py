from __future__ import annotations
import numpy as np
from typing import *
import random
import matplotlib.pyplot as plt
try:
    from utils.EnvironmentBasics import State, Action, Environment, StateOrSid, ActionOrAid
except:
    from EnvironmentBasics import State, Action, Environment, StateOrSid, ActionOrAid

class Policy():
    def __init__(self, environment: Environment, zero_init: bool=False) -> None:
        """
        Policy is one state to many actions with probabilities
        Basic Usage:
            policy[state or sid, action or aid] = value
            action_prob = policy[state or sid, action or aid]
        :param environment: the environment that the policy is for
        :param zero_init: whether to initialize the policy to all zeros
        """
        self.state_to_sid = environment.state_to_sid
        self.action_to_aid = environment.action_to_aid

        # Policy is one state to many actions with probabilities
        # policy[si, ai] = prob of taking action ai in state si
        self.__policy: np.ndarray = np.ones((environment.n_states, environment.n_actions)) * -np.inf
        for state in environment.states:
            if not state.is_terminal:
                valid_action_ids = environment.getValidActionIds(state)
                self.__policy[self.state_to_sid[state], valid_action_ids] = (1 / len(valid_action_ids)) if not zero_init else 0

        
    def __eq__(self, other: Policy) -> bool:
        return np.array_equal(self.__policy, other.__policy)

    def __repr__(self) -> str:
        return f"Policy({self.__policy})"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, indices: Union[Tuple[StateOrSid, ActionOrAid], int, State]) -> Union[float, np.ndarray]:
        """Usage: policy[state or sid, action or aid]"""
        if isinstance(indices, int):
            return self.__policy[indices]
        elif isinstance(indices, State):
            return self.__policy[self.state_to_sid[indices]]
        else:
            if isinstance(indices[0], State):
                indices = (self.state_to_sid[indices[0]], self.action_to_aid[indices[1]])
            if isinstance(indices[1], Action):
                indices = (indices[0], self.action_to_aid[indices[1]])
            return self.__policy[indices]

    def __setitem__(self, indices: Union[Tuple[StateOrSid, ActionOrAid], int, State], value: Union[float, np.ndarray]) -> None:
        """Usage: policy[state or sid, action or aid] = value"""
        # You cannot assign a value to a state-action pair if the action is not valid in that state
        if isinstance(indices, int):
            self.__policy[indices, self.__policy[indices] != -np.inf] = value
        elif isinstance(indices, State):
            self.__policy[self.state_to_sid[indices], self.__policy[self.state_to_sid[indices]] != -np.inf] = value
        else:
            if isinstance(indices[0], State):
                indices = (self.state_to_sid[indices[0]], self.action_to_aid[indices[1]])
            if isinstance(indices[1], Action):
                indices = (indices[0], self.action_to_aid[indices[1]])
            assert np.all(self.__policy[indices] != -np.inf), f"Action {indices[1]} is not valid in state {indices[0]}"
            self.__policy[indices] = value


    def diff(self, other: Policy) -> float:
        mask = self.__policy != -np.inf
        return float(np.sum(np.abs(self.__policy[mask] - other.__policy[mask])))


    def getValidActionIds(self, state: StateOrSid) -> np.ndarray:
        """Returns the action ids that are valid in the given state"""
        if isinstance(state, State):
            state = self.state_to_sid[state]
        return np.where(self.__policy[state] != -np.inf)[0]

    def normalize(self) -> None:
        """Normalizes the policy so that the sum of probabilities of all actions in a state is 1"""
        for state in range(self.__policy.shape[0]):
            valid_action_ids = np.where(self.__policy[state] != -np.inf)[0]
            self.__policy[state, valid_action_ids] /= np.sum(self.__policy[state, valid_action_ids])

    def rankNormalize(self) -> None:
        """Normalizes the policy so that the sum of probabilities of all actions in a state is 1"""
        for state in range(self.__policy.shape[0]):
            valid_action_ids = np.where(self.__policy[state] != -np.inf)[0]
            values = np.linspace(0.4, 1, len(valid_action_ids))
            rank = np.argsort(self.__policy[state, valid_action_ids])
            self.__policy[state, valid_action_ids[rank]] = values

    def isValidAction(self, state: StateOrSid, action: ActionOrAid) -> bool:
        """Returns True if the action is valid in the given state"""
        return self[state, action] != -np.inf
    

    def copy(self, environment) -> Policy:
        """Returns a copy of the policy"""
        new_policy = Policy(environment)
        new_policy.__policy = self.__policy.copy()
        return new_policy


class Agent():
    def __init__(self, environment: Environment, policy: Policy) -> None:
        """
        Agent should have a policy which tells what action to take given a state
        Agent should have a value function which tells the value of any state
        Agent should have a Q function which tells the value of any state-action pair
        :param environment: the environment that the agent is in
        :param policy: the policy that the agent follows
        """
        self.policy = policy
        self.state_value = np.zeros(environment.n_states)
        # although action_value_function is a Policy object, it is not a policy
        # It just has the same structure as a policy
        # Policy is one state to many actions with probabilities
        # Whereas action_value_function is one state to many actions with values
        self.action_value = Policy(environment, zero_init=True)

    def __repr__(self) -> str:
        return f"Agent({self.policy})"

    def __str__(self) -> str:
        return self.__repr__()

    def getPolicyActions(self, state: StateOrSid) -> np.ndarray:
        """Get the best actions at a given state following the current policy

        :param state: the given state
        :return: the best actions
        """
        best_action_prob = np.max(self.policy[state])
        return np.where(self.policy[state] == best_action_prob)[0]
    
    def getBestActions(self, state: StateOrSid) -> np.ndarray:
        """Get the best actions at a given state following the current action-value function

        :param state: the given state
        :return: the best actions
        """
        best_action_value = np.max(self.action_value[state])
        return np.where(self.action_value[state] == best_action_value)[0]

    def takeAction(self, state: StateOrSid, epsilon: float=0.1) -> Action:
        """
        Get an action at a given state following the current policy, with epsilon-greedy
        :param state: the given state
        :param epsilon: there is epsilon% chance of choosing a random action
        :return: the action to take
        """
        # epsilon-greedy
        if random.random() < epsilon:
            return np.random.choice(self.policy.getValidActionIds(state))
        else:
            return np.random.choice(self.getPolicyActions(state))
        

    def copy(self, environment) -> Agent:
        """Returns a copy of the agent"""
        new_agent = Agent(environment, self.policy)
        new_agent.state_value = self.state_value.copy()
        new_agent.action_value = self.action_value.copy(environment)
        return new_agent
