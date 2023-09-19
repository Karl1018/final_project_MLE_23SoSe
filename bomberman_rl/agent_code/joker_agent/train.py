from collections import namedtuple, deque

import pickle
import numpy as np
from typing import List

import events as e
from .callbacks import *
from .RewardCurve import *

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 2000

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
    self.reward_curve = RewardCurve()
    self.round = 0

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
    
    # Stores the current tansition
    self.model.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward_from_events(self, events)))
    self.model.learn_batched()

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
    #self.model.transitions.append(Transition(state_to_features(self, last_game_state), last_action, None, reward_from_events(self, events)))
    self.model.learn_batched()
    #
    self.model.target_network.load_state_dict(self.model.evaluation_network.state_dict())

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

    self.reward_curve.draw(self.round)
    self.round += 1


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,

        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -10,
        e.BOMB_DROPPED: -10,
        e.INVALID_ACTION: -20,
        
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
    self.reward_curve.record(reward_sum)
    return reward_sum


def calculate_n_step_td_updates(transitions: List[Transition], n: int, gamma: float) -> np.ndarray:
    """
    This function will calculate and apply n-step TD updates to the Q-values of each stored transition according to n-step TD Q-learning.

    :param transitions: Transition list with Q-values to be updated.
    :param n: Updates up to n steps in the future will be considered.
    :param gamma: Discount factor.

    :return: NumPy array containing the n-step TD updates for each transition.
    """
    n_step_updates = np.zeros(len(transitions))
    for t in range(len(transitions) - n, n - 1, -1):  # Iterate until the last n-1 transitions
        n_step_return = 0
        for i in range(n):
            n_step_return += (gamma ** i) * transitions[t + i].reward
        # Apply the n-step TD update to Q-values (assuming Q-values are available)
        n_step_updates[t] = n_step_return   
        return n_step_updates


#def calculate_n_step_rewards(transitions: List[Transition], n: int, gamma: float) -> None:
#     """
#     This function will calculate and then update the rewards of each stored transition (Transition.reward) according to n-step TD Q-learning.
            
#     :param transitions: Transition list to be updated.
#     :param n: Rewards up to n steps in the future will be considered.
#     :param gamma: Disconting factor.
    
#     :return: None
#     """
#     for t in range(len(transitions)):
#         n_step_reward = 0
#         for i in range(n):
#             if t + i < len(transitions):
#                 n_step_reward += (gamma ** i) * transitions[t + i].reward
#         transitions[t].reward = n_step_reward