import os
import pickle
import copy

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
    eps = linear_epsilon_decay(EPSILON_START, EPSILON_END, EPSILON_DECAY_DURATION, self.model.step)
    features = state_to_features(self, game_state)
    if self.model.step % LOG_EPISODE == 0:
        self.logger.debug(f"Current state: \n{features[0]}\n{features[1]}\n{features[2]}\n{features[3]}\n{features[4]}")
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

def destructible_crates_count(game_state: dict) -> int:
    """
    Calculates the number of crates taht will be destroyed if a bomb is dropped by the agent. Used to generate reward to encourage
    the agent to destroy more crates.

    Args:
        game_state (dict): State of game.

    Returns:
        int: Number of crates taht will be destroyed.
    """
    player_pos = np.array(game_state["self"][3])
    destructible_crates = 0
    for direction in np.array([[0,1], [0,-1], [1,0], [-1,0]]):
        for distance in range(1, 4):
            impact_subregion = direction * distance + player_pos
            pos = game_state['field'][impact_subregion[0], impact_subregion[1]]
            if pos == -1:
                break
            if pos == 1:
                destructible_crates += 1
    return destructible_crates

def features_with_coins(basic_field_map, game_state: dict) -> np.array:
    """
    This function provide a field map with basic information and the position of coins

    :basic_field_map: a field map with basic information
    :param game_state:  A dictionary describing the current game board.
    :return: the field map with the position of coins 
    """
    field_with_coins = basic_field_map

    # Put the position of coins into field (1)
    coins = game_state['coins']
    for coin in coins:
        coin_position = list(coin)
        field_with_coins[coin_position[0], coin_position[1]] = 5
    return field_with_coins

def features_with_bombs(basic_field_map, game_state: dict) -> np.array:
    """
    This function provide a field map with basic information and the position of coins

    :basic_field_map: a field map with basic information
    :param game_state:  A dictionary describing the current game board.
    :return: the field map with the position of enemy agent, bombs and explosion map 
    """
    field_with_bombs = basic_field_map

    bombs = game_state['bombs']
    for bomb in bombs:
        bomb_position = list(bomb)[0]
        bomb_timer = list(bomb)[1]
        blast_coords = []
        blast_coords = bomb.get_blast_coords(game_state['field'])
        for x in range(blast_coords.shape[0]):
            for y in range(blast_coords.shape[1]):
                field_with_bombs[x, y] = bomb_timer

    return field_with_bombs

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
    
    agent = game_state['self']
    agent_position = list(agent)[3]

    basic_map = game_state['field']
    
    empty_map = np.zeros(basic_map.shape)

    agent_map = copy.deepcopy(empty_map)
    agent_map[agent_position[0],agent_position[1]] = 10
    coin_map = features_with_coins(copy.deepcopy(empty_map), game_state)
    
    future_explosion_map = fill_explosion_map(np.array(game_state["explosion_map"]), game_state["bombs"], basic_map)
    
    enemy_map = copy.deepcopy(empty_map)
    other_agents = game_state['others']
    for other_agent in other_agents:
        other_agent_position = list(other_agent)[3]
        enemy_map[other_agent_position[0], other_agent_position[1]] = 10

    self.destructible_crates = destructible_crates_count(game_state)

    return np.array([basic_map[1:-1, 1:-1].T, coin_map[1:-1, 1:-1].T, future_explosion_map[1:-1, 1:-1].T, agent_map[1:-1, 1:-1].T, enemy_map[1:-1, 1:-1].T])

def linear_epsilon_decay(epsilon_start, epsilon_end, decay_duration, current_step, min_epsilon) -> float:
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