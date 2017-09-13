
import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greddy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greddy
        # define column name
        self.q_table = pd.DataFrame(columns=self.actions)



    def choose_action(self, observation):
        ''' chooose next action to execute'''
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # if two action got same value, we need to random choose one
            # random choose this two action
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action



    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)
        q_predict = self.q_table.ix[state, action]
        if state_ != 'terminal':
            q_target = reward + self.gamma * self.q_table.ix[state_, :].max()

        else:
            q_target = reward

        self.q_table.ix[state, action] += self.lr * (q_target - q_predict)



    def check_state_exist(self, state):
        # if not exist, add this state to q_table
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
