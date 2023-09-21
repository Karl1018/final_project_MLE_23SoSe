import matplotlib.pyplot as plt
import numpy as np

UPDATE_PERIOD = 50 # Period to update the figure.

class RewardRecorder():
    """
    Records total reward and scores each round and generate a figure to evaluate the behavior of the agent.
    """
    def __init__(self):
        self.rewards = []
        self.scores = []
        self.reward_counter = 0
        self.score_counter = 0

    def record_reward(self, reward: int):
        self.reward_counter += reward

    def record_score(self, score: int):
        self.score_counter += score

    def update(self, round: int):
        
        self.rewards.append(self.reward_counter)
        self.scores.append(self.score_counter)
        self.reward_counter = 0
        self.score_counter = 0
        reward_averages = []
        score_averages = []

        if round % UPDATE_PERIOD == 0:
            for i in range(0, len(self.rewards), UPDATE_PERIOD):

                chunk_r = self.rewards[i:i+UPDATE_PERIOD]
                average_r = sum(chunk_r) / len(chunk_r)
                reward_averages.append(average_r)

                chunk_s = self.scores[i:i+UPDATE_PERIOD]
                average_s = sum(chunk_s) / len(chunk_s)
                score_averages.append(average_s)
            
            plt.clf()
            x = np.arange(0, len(chunk_r), UPDATE_PERIOD)
            # Draws the average rewards over each period.
            fig, ax1 = plt.subplots()
            ax1.plot(x, reward_averages, color='b', label='reward')
            ax1.set_xlabel('iterations')
            ax1.set_ylabel('reward')
            ax1.legend()
            # Draws the score rewards over each period.
            ax2 = ax1.twinx()
            ax2.plot(x, score_averages, color='r', label='score')
            ax2.set_ylabel('score')
            ax2.legend()

            plt.title('Reward & Score Curve')
            plt.savefig(f"logs/reward_curve.png", bbox_inches='tight', pad_inches=0.4)