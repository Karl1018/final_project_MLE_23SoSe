import numpy as np

def explosion_map(field) -> np.array:
    """
    n steps to explode: -n-1
    one turn after explosion: -1

    Args:
        field: field of the game

    Returns:
        np.array: explosion map
    """
    return
def safty_map(field) -> np.array:
    """
    Adds the free tiles with the explosion map. Nagative slots stands for danger.

    Args:
        field: field of the game

    Returns:
        np.array: safty map
    """
    return
def distance_to_coin(field, coins, agent) -> np.array:
    """
    Calculates distances to to the nearest coins from 4 directions.

    Args:
        field: field of the game
        coins: positions of coins
        agent: positions of coins

    Returns:
        np.array: 4*1 vector representing the distance to the nearest coins from 4 directions.
    """
    return
def destroyable_crates(field, agent) -> int:
    """
    Calculates the number of crates that would be destroyed if a bomb is dropped here.

    Args:
        field: field of the game

    Returns:
        int: the number of crates that would be destroyed
    """
    return
def check_safety(field) -> np.ndarray:
    """
    Checks for all 4 dicretions if it would be dangerous for the agent

    Args:
        field: field of the game

    Returns:
        np.ndarray: 4*1 vector
    """
    return