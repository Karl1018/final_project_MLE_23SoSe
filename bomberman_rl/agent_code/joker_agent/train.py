from collections import namedtuple, deque

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import *

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 1000

# Events
CREATES_TO_DESTROY = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, s', r)
    self.model.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    """r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    r_new = r1 + r2"""
    # Stores the current tansition
    self.model.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events)))
    self.model.learn()

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.model.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))
    self.model.learn()
    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,

        e.MOVED_RIGHT: 0,
        e.MOVED_LEFT: 0,
        e.MOVED_UP: 0,
        e.MOVED_DOWN: 0,
        e.WAITED: -10,
        e.BOMB_DROPPED: -1,
        e.INVALID_ACTION: -200,
        
        e.KILLED_SELF: -500,
        e.GOT_KILLED: -500,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if e.BOMB_DROPPED in events:
            reward_sum += self.destructible_crate * 50

    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def calculate_n_step_rewards(transitions: List[Transition], n: int, gamma: float) -> None:
    """
    This function will calculate and then update the rewards of each stored transition (Transition.reward) according to n-step TD Q-learning.
            
    :param transitions: Transition list to be updated.
    :param n: Rewards up to n steps in the future will be considered.
    :param gamma: Disconting factor.
    
    :return: None
    """
    n_step_reward = 0
    n_step_rewards_list = [] 
    for t in range(len(transitions)):
        for i in range(n):
            if t + i < len(transitions):
                n_step_reward += (gamma ** i) * transitions[t + i].reward
        n_step_rewards_list.append(n_step_reward)
    return n_step_rewards_list
