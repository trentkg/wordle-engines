import gym
import gym_wordle
import itertools
import numpy as np
import pandas as pd
import sys
import wordle
import bisect
import pickle
import matplotlib.pyplot as plt
from os.path import exists
 
from collections import defaultdict

# Number of valid guesses in wordle
WORDLE_ACTION_SPACE_SIZE = 12972

def create_blank_state():
    return np.zeros(WORDLE_ACTION_SPACE_SIZE)

class QLearningAlgorithm:
    def __init__(self, state_file='artifacts/qlearning.pkl', epsilon = 0.1):
        self.state_file = state_file
        self.policy = self.create_epsilon_greedy_policy(epsilon, WORDLE_ACTION_SPACE_SIZE)
        if exists(state_file):
            with open(state_file, mode='rb') as f:
                self.q_matrix = pickle.load(f) 
        else:
            self.q_matrix = defaultdict(create_blank_state)

    def save_state(self, state_file):
        with open(state_file, mode='wb') as f:
            pickle.dump(self.q_matrix, f)

    @classmethod
    def train(cls, num_episodes, state_file='artifacts/qlearning.pkl'):
        env = gym.make("Wordle-v0")

        algo = QLearningAlgorithm(state_file)

        q_matrix = algo.qlearn(env, num_episodes)
        algo.save_state(state_file)

    def predict(self, state):
        action_probabilities = self.policy(self.q_matrix, state)
       
        action = np.random.choice(np.arange(
                  len(action_probabilities)),
                   p = action_probabilities)
        # i.e. a word. 
        return action

    def create_epsilon_greedy_policy(self, epsilon, num_actions):
        def policy_function(q_matrix, state):

            action_probabilities = np.ones(num_actions,
                    dtype = float) * epsilon / num_actions
            choices = q_matrix[state.tobytes()]
            # previously we used the following code to find the best action:
            # best_action = np.argmax(q_matrix[state.tobytes()])
            # but argmax provides the 0th index item in the case when all are ties.
            # this means when the algorithm is starting out, there is a heavy bias towards
            # the first choice, which is bad. so instead in the case of ties, 
            # we choose a random element from the list of ties.
            best_action = np.random.choice(np.where(choices == choices.max())[0])
            action_probabilities[best_action] += (1.0 - epsilon)
            return action_probabilities

        return policy_function

    def qlearn(self, env, num_episodes, discount_factor = 1.0,
                                alpha = 0.6):

        for ith_episode in range(num_episodes):
               
            state = env.reset()
               
            for t in itertools.count():

                action = self.predict(state)
        
                next_state, reward, done, _ = env.step(action)
                if done and reward == 0: # we lost
                    reward = -1

                best_next_action = np.argmax(self.q_matrix[next_state.tobytes()])    
                td_target = reward + discount_factor * self.q_matrix[next_state.tobytes()][best_next_action]
                td_delta = td_target - self.q_matrix[state.tobytes()][action]
                self.q_matrix[state.tobytes()][action] += alpha * td_delta
       
                if done:
                    if reward > 0:
                        print("we won!")
                    break
                       
                state = next_state
        return self.q_matrix

