import os
import pickle
import random

import numpy as np


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
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
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
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")
    return np.random.choice(ACTIONS, p=self.model)


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # Put the bombs into field
    bombs = game_state['bombs']
    field_with_bombs = game_state['field']
    for bomb in bombs:
        bomb_position = list(bomb)[0]
        field_with_bombs[bomb_position[0], bomb_position[1]] = -2
    
    return field_with_bombs

def get_field(game_state: dict, field_with_bombs) -> np.array:
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

