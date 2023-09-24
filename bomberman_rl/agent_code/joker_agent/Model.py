from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim

# Hyper parameters
LOG_EPISODE = 50 # Period to log.
TRANSITION_HISTORY_SIZE = 4000 # Memory size for expericence replay.
BATCH_SIZE = 1000
LEARNING_RATE = 0.05
GAMMA = 0.9 # Discounting factor.
N = 4 # N-step reward.
EPSILON_DECAY_DURATION = 1000
EPSILON_START = 1
EPSILON_END = 0.02

INDEX_ACTIONS = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'WAIT': 4, 'BOMB': 5}
NO_STATE_PLACEHOLDER = np.zeros((5, 9, 9)) # Representing next_state at the end of round.

device = ("cuda"
    if torch.cuda.is_available()
    else "cpu")

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # Layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(10, 6)

    def forward(self, x):
        batch_size = x.shape[0]
        handcrafted_features = x[:, -1, 0, :2].view(batch_size, 2)
        x = nn.functional.relu(self.conv1(x[:, :-1, :, :]))
        x = x.view(batch_size, -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = torch.cat((x, handcrafted_features), dim=1)
        return self.fc3(x)

class DQN():

    def __init__(self):
        # Initializes two networks.
        self.evaluation_network, self.target_network = Network(), Network()
        self.evaluation_network.to(device)
        self.target_network.to(device)
        self.target_network.load_state_dict(self.evaluation_network.state_dict())

        self.optimizer = torch.optim.Adam(self.evaluation_network.parameters(), lr=LEARNING_RATE)
        self.loss_func = nn.MSELoss()

        self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
        self.step = 0 # Time step.
        self.future_rewards = 0 # Remaining steps to go forward in n-step reward.

    def learn_batched(self):

        if len(self.transitions) < BATCH_SIZE:
            return

        # Experience replay.
        sample_indices = np.random.choice(min(len(self.transitions), TRANSITION_HISTORY_SIZE) - N, BATCH_SIZE)
        batch = [self.transitions[i] for i in sample_indices]
        states = np.array([sample.state for sample in batch], dtype=np.float32)
        actions = np.array([INDEX_ACTIONS[sample.action] for sample in batch], dtype=np.int64)
        next_states = np.array([sample.next_state for sample in batch], dtype=np.float32)
        rewards = np.array([sample.reward for sample in batch], dtype=np.float32)
        
        # Convers data to tensor and sends to GPU.
        states = torch.tensor(states).to(device)
        next_states = torch.tensor(next_states).to(device)
        with torch.no_grad():
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)

        Q_evaluate = self.evaluation_network(states).gather(1, actions.unsqueeze(1))
        # Finds transitions from the last steps of a game.
        equality_mask = torch.all(next_states == torch.tensor(NO_STATE_PLACEHOLDER, device=device).unsqueeze(0), dim=3)
        equality_mask = torch.all(equality_mask, dim=2)
        equality_mask = equality_mask[:, 0]
        equality_mask = torch.all(next_states == torch.tensor(NO_STATE_PLACEHOLDER, device=device).unsqueeze(0))

        Q_next = torch.where(equality_mask, 0, self.target_network(next_states).detach().max())
        Q_target = (rewards + GAMMA**(self.future_rewards + 1) * Q_next).unsqueeze(1)

        loss = self.loss_func(Q_evaluate, Q_target)
        loss.requires_grad_(True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step += 1


    def get_action(self, x):

        with torch.no_grad():
            x = torch.Tensor(x).unsqueeze(0).to(device)
            actions_value = self.evaluation_network.forward(x).cpu()
        
        return actions_value
    
    def update_n_step_td_reward(self, current_reward, round_ended = False) -> None:
        """
        This function will calculate and apply n-step TD updates to the Q-values of 
        each stored transition according to n-step TD Q-learning.

        :param transitions: Transition list with Q-values to be updated.
        :param n: Updates up to n steps in the future will be considered.
        :param gamma: Discount factor.
        """
        for i in range(1, self.future_rewards + 1):
            reward = self.transitions[-i].reward
            reward += (GAMMA ** i) * current_reward
            self.transitions[-i] = self.transitions[-i]._replace(reward = reward)
        self.future_rewards = min(self.future_rewards + 1, N)

        if round_ended:
            self.future_rewards = 0