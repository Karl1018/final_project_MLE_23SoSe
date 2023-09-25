from collections import namedtuple
from typing import List

import pickle
import events as e

from .callbacks import *
from .RewardRecorder import RewardRecorder

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, s', r)
    
    self.reward_recorder = RewardRecorder()
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
    if self.model.step % LOG_EPISODE == 0:
        self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    
    # Stores the current tansition
    reward = reward_from_events(self, events)
    self.model.update_n_step_td_reward(reward)
    self.model.transitions.append(Transition(state_to_features(self, old_game_state), self_action, state_to_features(self, new_game_state), reward))
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
    reward = reward_from_events(self, events)
    self.model.update_n_step_td_reward(reward, True)
    self.model.transitions.append(Transition(state_to_features(self, last_game_state), last_action, NO_STATE_PLACEHOLDER, reward))
    self.model.learn_batched()
    # Updates the target network at the end of each round.
    self.model.target_network.load_state_dict(self.model.evaluation_network.state_dict())
    self.reward_recorder.update(self.round)
    self.round += 1

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
        e.BOMB_DROPPED: 0,
        e.INVALID_ACTION: -20,
        
        e.KILLED_SELF: -50,
        e.GOT_KILLED: -50,
    }
    reward_sum = 0
    score_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
        if e.CRATE_DESTROYED in events:
            reward_sum += events.count('CRATE_DESTROYED') * 5 # Reward for destroying crates.
        if e.COIN_COLLECTED in events:
            score_sum += 1
    if self.model.step % LOG_EPISODE == 0:
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    # Records rewards and scores for evaluation.
    self.reward_recorder.record_reward(reward_sum)
    self.reward_recorder.record_score(score_sum)
    
    return reward_sum
