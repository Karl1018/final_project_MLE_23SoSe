import os
import pickle
import random

import numpy as np
import torch

from .Model import *


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

    eps = linear_epsilon_decay(1, 0.1, EPSILON_DECAY_DURATION, self.model.step)
    features = state_to_features(self, game_state)
    self.logger.debug(f"Current state: \n{features}")
    if np.random.rand() >= eps:
        actions_value = self.model.get_action(features)
        self.logger.debug(f"Output from model: {actions_value}")
        action = ACTIONS[torch.argmax(actions_value)]
        self.logger.debug(f"Query model for action: {action}")
    else:
        action = ACTIONS[np.random.randint(0, len(ACTIONS))]
        self.logger.debug(f"Choose action at random: {action} with epsilon {eps}")
    return action

def destructible_crate_count(game_state: dict) -> int:
    player_pos = np.array(game_state["self"][3])
    destructible_crate = 0
    for direction in np.array([[0,1], [0,-1], [1,0], [-1,0]]):
        for distance in range(1, 4):
            impact_subregion = direction * distance + player_pos
            pos = get_field(game_state)[impact_subregion[0], impact_subregion[1]]
            if pos == -1:
                break
            if pos == 1:
                destructible_crate += 1
    return destructible_crate

# agent:6; other_agent:-6; coin:8; bomb:-8
def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: the field map with the position of agents 
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    basic_field_map = game_state['field']
    
    # Put the position of agent into field (6)
    agent = game_state['self']
    agent_position = list(agent)[3]
    basic_field_map[agent_position[0],agent_position[1]] = 6

    # Split the boundaries in the field
    basic_field_map = basic_field_map[1:-1, 1:-1]
    
    #field = get_field(game_state)
    self.destructible_crate = destructible_crate_count(game_state)

    return basic_field_map

def features_with_coins(self, basic_field_map, game_state: dict) -> np.array:
    """
    This function provide a field map with basic information and the position of coins

    :basic_field_map: a field map with basic information
    :param game_state:  A dictionary describing the current game board.
    :return: the field map with the position of coins 
    """
    field_with_coins = basic_field_map

    # Put the position of coins into field (8)
    coins = game_state['coins']
    for coin in coins:
        coin_position = list(coin)
        field_with_coins[coin_position[0], coin_position[1]] = 8

    return field_with_coins

def features_with_bombs(self, basic_field_map, game_state: dict) -> np.array:
    """
    This function provide a field map with basic information and the position of coins

    :basic_field_map: a field map with basic information
    :param game_state:  A dictionary describing the current game board.
    :return: the field map with the position of bombs and explosion map 
    """
    field_with_bombs = basic_field_map

    # Put the position of enemy agents
    other_agents = game_state['others']
    for other_agent in other_agents:
        other_agent_position = list(other_agent)[3]
        field_with_bombs[other_agent_position[0], other_agent_position[1]] = -6

    # Put the position of bombs into field (-8)
    bombs = game_state['bombs']
    for bomb in bombs:
        bomb_position = list(bomb)[0]
        field_with_bombs[bomb_position[0], bomb_position[1]] = -8

    # Put the explosion range into field (-3)
    explosion_map = game_state['explosion_map']
    for x in range(explosion_map.shape[0]):
        for y in range(explosion_map.shape[1]):
            countdown_timer = explosion_map[x, y]
            if countdown_timer > 0:
                field_with_bombs[x, y] = countdown_timer

    return field_with_bombs

def get_cropped_field(game_state: dict, field_with_bombs) -> np.array:
    agent = game_state['self']
    agent_position = list(agent)[3]
    agent_pos_x = agent_position[0]
    agent_pos_y = agent_position[1]

    # Calculate the coordinates of the upper left and lower right corners of the cut area
    left_top_x = max(0, agent_pos_x - 2)
    left_top_y = max(0, agent_pos_y - 2)
    right_bottom_x = min(field_with_bombs.shape[1] - 1, agent_pos_x + 2)
    right_bottom_y = min(field_with_bombs.shape[0] - 1, agent_pos_y + 2)

    sliced_field = field_with_bombs[left_top_y:right_bottom_y + 1, left_top_x:right_bottom_x + 1]

    return sliced_field

def get_field(game_state: dict) -> np.array:
    # Put the bombs into field
    bombs = game_state['bombs']
    field_with_bombs = game_state['field']
    for bomb in bombs:
        bomb_position = list(bomb)[0]
        field_with_bombs[bomb_position[0], bomb_position[1]] = -2

    return field_with_bombs

def linear_epsilon_decay(epsilon_start, epsilon_end, decay_duration, current_step, min_epsilon=0.05): #TODO: Hyperparameterize min_epsilon
    """

    Args:


    Returns:
        
    """
    decay_rate = (epsilon_start - epsilon_end) / decay_duration
    epsilon = max(epsilon_start - decay_rate * current_step, min_epsilon)
    return epsilon