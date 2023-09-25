import os
import pickle
import copy

import numpy as np
import torch

from .Model import *
from .FeatureEngineering import *


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.model = DQN()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    eps = linear_epsilon_decay(EPSILON_START, EPSILON_END, EPSILON_DECAY_DURATION, self.model.step)
    features = state_to_features(self, game_state)
    if self.model.step % LOG_EPISODE == 0 or not self.train:
        self.logger.debug(f"Current state: \n{features[0]}\n{features[1]}\n{features[2]}\n{features[3]}")
        #self.logger.debug(f"Current state: \n{features}")
    # Epsilon greedy strategy.
    if np.random.rand() >= eps:
        actions_value = self.model.get_action(features)
        action = ACTIONS[torch.argmax(actions_value)]
        if self.model.step % LOG_EPISODE == 0 or not self.train:
            self.logger.debug(f"Output from model: {actions_value}")
            self.logger.debug(f"Query model for action: {action}")
    else:
        action = ACTIONS[np.random.randint(0, len(ACTIONS))]
        if self.model.step % LOG_EPISODE == 0 or not self.train:
            self.logger.debug(f"Choose action at random: {action} with epsilon {eps}")
    return action

def fill_explosion_map(explosions, bombs, field):
    '''
    fill a explosion map with the bombs that are going to explode
    and updates the field array

    :param explosions: explosion map
    :param bombs: numpy array of current bombs in format ((x,y), t)
    :param field: numpy array of the field

    return: explosion map with timer as numpy array
    '''
    future_explosion_map = (np.copy(explosions)*-4) + 1 # -3 now exploding, 1 no bomb in reach
    for bomb in bombs:
        pos = np.array(bomb[0])
        timer = bomb[1] - 3 # the smaller, the more dangerous
        field[pos[0], pos[1]] = -2 # put the bombs in the field array as n obstackles

        for direction in np.array([[0,1], [0,-1], [1,0], [-1,0]]):
            for length in range(0, 4):
                beam = direction*length + pos
                obj = field[beam[0], beam[1]]
                if obj == -1:
                    break
                if future_explosion_map[beam[0], beam[1]] > timer:
                    future_explosion_map[beam[0], beam[1]] = timer

    return future_explosion_map



def linear_epsilon_decay(epsilon_start, epsilon_end, decay_duration, current_step, min_epsilon=0.02) -> float:
    """
    Generates the epilon with linear decay for greedy training strategy.

    Args:
        epsilon_start
        epsilon_end
        decay_duration
        current_step
        min_epsilon

    Returns:
        float: the epsilon for greedy training strategy.
    """ 
    decay_rate = (epsilon_start - epsilon_end) / decay_duration
    epsilon = max(epsilon_start - decay_rate * current_step, min_epsilon)
    return epsilon