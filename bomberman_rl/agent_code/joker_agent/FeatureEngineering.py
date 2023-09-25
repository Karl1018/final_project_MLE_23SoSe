import numpy as np
from collections import deque
#from .callbacks import ACTIONS



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

def get_coin_map(basic_field_map, game_state: dict) -> np.array:
    """
    This function provide a field map with basic information and the position of coins

    :basic_field_map: a field map with basic information
    :param game_state:  A dictionary describing the current game board.
    :return: the field map with the position of coins 
    """
    coin_map = basic_field_map

    # Put the position of coins into field (1)
    coins = game_state['coins']
    for coin in coins:
        coin_position = list(coin)
        coin_map[coin_position[0], coin_position[1]] = 5
    return coin_map

def get_explosion_map(field, bombs, pos=None):
    """
    This function provide a field map with basic information and the position of bombs

    :basic_field_map: a field map with basic information
    :param game_state:  A dictionary describing the current game board.

    :return: the field map with the explosion range 
    """

    explosion_map = np.zeros_like(field)

    # Put the explosion range into field 
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if pos is None:
        for bomb in bombs:
            bomb_position = np.array(bomb[0])
            bomb_timer = np.array(bomb[1])
            explosion_map[bomb_position[0], bomb_position[1]] = bomb_timer + 2
            # x, y = bomb_position
            # Iterate over the directions
            for dx, dy in directions:
                # Initialize the current position
                current_position_x, current_position_y = bomb_position
                # Iterate in the direction
                for _ in range(4): 
                    # Calculate the next position
                    next_position_x, next_position_y = current_position_x + dx, current_position_y + dy
                    # Check if the next position is within the field boundaries
                    if 0 <= next_position_x < explosion_map.shape[0] and 0 <= next_position_y < explosion_map.shape[1]:
                        # Check if the next position is a wall (-1)
                        if field[next_position_x, next_position_y] != 0:
                            break 
                        else:
                            explosion_map[next_position_x, next_position_y] = bomb_timer + 2
                            # Move to the next position
                            current_position_x, current_position_y = next_position_x, next_position_y
                    else:
                        break  # Stop if outside of boundaries
    else:
        bomb_position = pos
        bomb_timer = 3
        explosion_map[bomb_position[0], bomb_position[1]] = bomb_timer + 2
        # x, y = bomb_position
        # Iterate over the directions
        for dx, dy in directions:
            # Initialize the current position
            current_position_x, current_position_y = bomb_position
            # Iterate in the direction
            for _ in range(4): 
                # Calculate the next position
                next_position_x, next_position_y = current_position_x + dx, current_position_y + dy
                # Check if the next position is within the field boundaries
                if 0 <= next_position_x < explosion_map.shape[0] and 0 <= next_position_y < explosion_map.shape[1]:
                    # Check if the next position is a wall (-1)
                    if field[next_position_x, next_position_y] == -1:
                        break 
                    else:
                        explosion_map[next_position_x, next_position_y] = bomb_timer + 2
                        # Move to the next position
                        current_position_x, current_position_y = next_position_x, next_position_y
                else:
                    break  # Stop if outside of boundaries
    return explosion_map

def get_safety_map(field, agent_position) -> np.array:
    """
    Adds the free tiles with the explosion map. Negative slots stand for danger.

    Args:
        field: field of the game

    Returns:
        np.array: safety map
    """
    # Create an explosion map using the explosion_map function
    exp_map = get_explosion_map(field, None, agent_position)
    safety_map = np.copy(field)

    safety_map[safety_map == 0] = exp_map[safety_map == 0]
    #print(safety_map.T)
    return safety_map

def crop_map(field, agent_position, padding) -> np.ndarray:
    rows, cols = field.shape

    cropped_region = np.full((9, 9), padding)
        
    top_left = (agent_position[0] - 4, agent_position[1] - 4)

    for j in range(9):
        for i in range(9):
            x, y = top_left[0] + i, top_left[1] + j
            if 0 <= x < rows and 0 <= y < cols:
                cropped_region[i, j] = field[x, y]
        
    return cropped_region



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
    
    field = game_state['field']
    bombs = [(bomb_pos, bomb_timer) for (bomb_pos, bomb_timer) in game_state['bombs']]
    _, _, bombs_left, agent_position = game_state['self']
    empty_map = np.zeros(field.shape)

    basic_map = np.copy(field)
    agent_map = np.copy(empty_map)
    agent_map[agent_position[0],agent_position[1]] = 1
    coin_map = get_coin_map(np.copy(empty_map), game_state)
    explosion_map = get_explosion_map(field, bombs)
    enemy_map = np.copy(empty_map)
    other_agents = game_state['others']
    for other_agent in other_agents:
        other_agent_position = list(other_agent)[3]
        enemy_map[other_agent_position[0], other_agent_position[1]] = 1
    safety_map = get_safety_map(field, agent_position)
    #Maps for CNN
    basic_map = crop_map(basic_map, agent_position, -1)
    agent_map = crop_map(agent_map, agent_position, 0)
    coin_map = crop_map(coin_map, agent_position, 0)
    explosion_map = crop_map(explosion_map, agent_position, 0)
    enemy_map = crop_map(enemy_map, agent_position, 0)
    #Hand-crafted
    #safety_map = crop_map(safety_map, agent_position, -1)
    directions = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    def is_deadend(pos, safety_map, turns=0):
        queue = deque()
        visited = []
        queue.append((list(pos), turns))
        while len(queue):
            pos, turns = queue.popleft()

            if turns > 4:
                break
            if pos in visited:
                continue
            if turns - 1 + safety_map[pos[0], pos[1]] == 5:
                continue
            if safety_map[pos[0], pos[1]] == 0:
                #print(f"Found way out at {(pos[0], pos[1])}")
                return False
            visited.append(pos)
            neighborhood = []
            #print(pos, (pos + directions))
            for p in (pos + directions):
                if field[p[0], p[1]] == 0:
                    neighborhood.append(list(p))
            for neighbor in neighborhood:
                queue.append((neighbor, turns+1))
                
        return True

    def get_valid_actions():
        x = agent_position[0]
        y = agent_position[1]
        neighborhood = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y), (x, y)]

        validity = np.zeros(5)

        for i, neighbor in enumerate(neighborhood):
            validity[i] = 1 if (field[neighbor] == 0 and not neighbor in other_agents) else 0
        validity[4] = bombs_left
        return validity

    destructible_crates = destructible_crates_count(game_state)
    validity = get_valid_actions()
    #safety_info = check_safety(np.copy(field), agent_position, bombs)
    #, np.concatenate([destructible_crates], safety_info)
    handcrafted_map = np.zeros_like(basic_map)
    handcrafted_map[0, 0] = destructible_crates
    for i, item in enumerate(validity):
        handcrafted_map[0, i + 1] = item
    #handcrafted_map[0, 5] = is_deadend((4, 4), safety_map)
    return np.array([basic_map, coin_map, explosion_map, handcrafted_map])

























def distance_to_coin(field, coins, agent) -> np.array:
    """
    Calculates distances to the nearest coins from 4 directions.

    Args:
        field: field of the game
        coins: positions of coins
        agent: position of the agent

    Returns:
        np.array: 4*1 vector representing the distance to the nearest coins from 4 directions.
    """
    # Define directions (UP, RIGHT, DOWN, LEFT)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Initialize distances to maximum values
    distances = np.full(4, np.inf)

    for i, (dx, dy) in enumerate(directions):
        x, y = agent
        while True:
            x += dx
            y += dy
            if x < 0 or x >= field.shape[0] or y < 0 or y >= field.shape[1] or field[x, y] == -1:
                break  # Stop if we hit a wall or go out of bounds
            if (x, y) in coins:
                distance = abs(x - agent[0]) + abs(y - agent[1])  # Manhattan distance
                distances[i] = min(distances[i], distance)
                break

    return distances

def check_safety(field, agent_pos, bombs) -> np.ndarray:
    """
    Checks for all 4 directions if it would be dangerous for the agent

    Args:
        field: field of the game

    Returns:
        np.ndarray: 4*1 vector
    """
    # Define directions (UP, RIGHT, DOWN, LEFT)
    directions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    neighborhood = directions + np.array([agent_pos[0], agent_pos[1]])
    neighborhood = np.clip(neighborhood, 0, 16)
    safety_vector = np.zeros(4, dtype=bool)

    safety_map = get_safety_map(field, bombs)

    for i, pos in enumerate(neighborhood):
        if safety_map[pos[0], pos[1]] < 0 or safety_map[pos[0], pos[1]] == 1:
            safety_vector[i] = False
        else:
            safety_vector[i] = True
    return safety_vector

def state_to_features_abandoned(self, game_state: dict) -> np.array:
    """
    Converts the game state to a feature vector.

    Args:
        game_state: A dictionary describing the current game board.
        coordinate_history: A deque containing the agent's coordinate history.

    Returns:
        np.array: A feature vector.
    """

    if game_state is None:
        return None

    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']
    bombs = [(bomb_pos, bomb_timer) for (bomb_pos, bomb_timer) in game_state['bombs']]

    # Calculate distance to nearest coins in four directions
    distances_to_coins = distance_to_coin(np.copy(field), coins, (x, y))

    # Calculate the number of destroyable crates
    #crates_destroyed = destroyable_crates(field, (x, y))
    self.destructible_crates = destructible_crates_count(game_state)

    # Calculate safety information
    #safety_info = check_safety(np.copy(field), (x, y), bombs)

    # Combine all features into a single array
    #print(safety_map)
    features = np.concatenate((distances_to_coins, [self.destructible_crates], safety_info, [bombs_left]))
    #print(features.shape)
    return features