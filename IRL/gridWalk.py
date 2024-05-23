# _*_ coding : utf-8 _*_
# @Time : 2024-03-23 9:31
# @Author : PeterPan
# @File : gridWalk
# @Project : maxent_gridworld.py
import gym
import gym.spaces as spaces
import numpy as np
import math
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True, threshold=5000)

class bicycleGridRiding(gym.Env):

    def __init__(self,
                 nrow=92,
                 ncol=47,
                 moveProb=0.5,
                 defaultReward=-0.2,
                 attrTable=pd.DataFrame()):
        '''
        Attention:  Alias Sampling?
        Grid: Matrix, size=(m,n,p)
        m : the number of rows in environment
        n : the number of columns in environment
        p : the size of feature vector of states in environment
        :param grid: matrix contains location info
        :param moveProb: default probility of random walk
        :param defaultReward: default reward of each step while working
        '''
        self.nrow = nrow
        self.ncol = ncol
        self.moveProb = moveProb
        self.attrTable = attrTable

        # Action Space
        self.action_space = spaces.Discrete(9, start=-4)
        self.observation_space = spaces.Dict({
            # 'agentLocation': spaces.Box(low=0, high=max(self.nrow-1, self.ncol-1), shape=(2,), dtype=int),
            'agentLocation': spaces.Discrete(len(self.states)),
            'agentTarget': spaces.Discrete(len(self.states)),
            'BuildEnv': spaces.Dict({
                'RDNorm': spaces.Box(low=0, high=1, dtype=int),
                'Pass': spaces.Discrete(2, start=0)
            })
        })

        # Action Space: Dictionary
        self._actions = {
            # opposite_direction = (action + 2) % 8
            -1: "LEFT",
            2: "DOWN",
            1: "RIGHT",
            -2: "UP",
            3: "UPLEFT",
            4: "DOWNLEFT",
            -3: "DOWNRIGHT",
            -4: "UPRIGHT",
            0: "STAY"
        }

        self.defaultReward = defaultReward
        self.moveProb = moveProb

    @property
    def states(self):
        return list(range(self.nrow * self.ncol))

    def coordinate_to_state(self, row, col):
        '''
        Take [91, 0] As An Example
        Take [0,  0] As An Example
        47 * 92
        '''
        index = (self.nrow - (row + 1)) * self.ncol + col
        return index

    def state_to_coordinate(self, state):
        '''
        [91, 0] -> 0
        [0,  0] -> 4377
        index - col = (92-(row+1)) * 47
        '''
        row = self.nrow - math.floor(state / self.ncol) - 1
        a = (self.nrow - (row + 1)) * self.ncol
        column = state - a
        return row, column

    def action2Index(self, a):
        index = list(self._actions.keys()).index(a)
        return index

    def index2Action(self, index):
        action = list(self._actions.keys())[index]
        return action

    def state_to_feature(self, s):
        '''
        One - Hot Encoding
        '''
        # feature = np.zeros(self.nrow * self.ncol)    # [ObservtionSpace, 1]
        # feature[s] = 1.0
        feature = self.attrTable.iloc[s].to_numpy()[1:]
        return feature

    def _get_obs(self):
        state = self._agent_location
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "roadDense": self.attrTable['roadDense'][state],
            "Road":self.attrTable['Road'][state],
            "Build":self.attrTable['Build'][state],
            "Vegetation":self.attrTable['Vegetation'][state],
            "wall":self.attrTable['wall'][state],
            "Fence":self.attrTable['Fence'][state],
            "Pole":self.attrTable['Pole'][state],
            "TrafLight":self.attrTable['TrafLight'][state],
            "TrafSig":self.attrTable['TrafSig'][state],
            "Terrain":self.attrTable['Terrain'][state],
            "Sky":self.attrTable['Sky'][state],
            "Person":self.attrTable['Person'][state],
            "Rider":self.attrTable['Rider'][state],
            "Car":self.attrTable['Car'][state],
            "Truck":self.attrTable['Truck'][state],
            "Bus":self.attrTable['Bus'][state],
            "Train":self.attrTable['Train'][state],
            "MotorCycle":self.attrTable['MotorCycle'][state],
            "Bicycle":self.attrTable['Bicycle'][state],
            "Sidewalk":self.attrTable['Sidewalk'][state],
            "Slope":self.attrTable['Slope'][state],
            "NDVI":self.attrTable['NDVI'][state],
            "POI":self.attrTable['POI'][state]
                }

    def _get_info(self):
        agentLoc = np.array(self.state_to_coordinate(self._agent_location))
        targetLoc = np.array(self.state_to_coordinate(self._target_location))
        return {"distance": np.linalg.norm(agentLoc - targetLoc, ord=1)}

    def transitFunc(self, state, action):
        '''
        Dict Version
        P = {
         state 1 : {next State 1 : P11, next State 2 : P12, ···},
         state 2 : {next State 1 : P21, next State 2 : P22, ···},
        }
        P = [
         [P11, P12, P13]
        ]
        '''
        transition_probs = {}
        opposite_direction = -action  # Input: Action Index
        
        if action != 0:
            candidates = [a for a in list(range(-4, 5, 1))
                          if a != opposite_direction]  # a != 0 : 8 个选择
            for a in candidates:
                if a == action:
                    prob = self.moveProb
                else:
                    prob = (1 - self.moveProb) / 7

                next_state = self._move(state, a)    # Whether It Is the Correct Next State
                if next_state not in transition_probs:
                    transition_probs[next_state] = prob
                else:
                    transition_probs[next_state] += prob

        else:
            candidates = [a for a in list(range(-4, 5, 1))]
            for a in candidates:
                if a == action:
                    prob = self.moveProb
                else:
                    prob = (1 - self.moveProb) / 8

                next_state = self._move(state, a)
                if next_state not in transition_probs:
                    transition_probs[next_state] = prob
                else:
                    transition_probs[next_state] += prob

        return transition_probs

    def transitFuncArray(self, state, action):
        '''
        Array Version
        State:[
            Action[PS1, PS2, ···, PSn]
        ]
        '''
        transition_probs = np.zeros((len(self.states),))    # [numStates,]
        opposite_direction = -action  # Input: Action Index
        
        if action != 0:
            candidates = [a for a in list(range(-4, 5, 1))
                          if a != opposite_direction]  # a != 0 : 8 个选择
            for a in candidates:
                if a == action:
                    prob = self.moveProb
                else:
                    prob = (1 - self.moveProb) / 7

                next_state = self._move(state, a)  # Whether It Is the Correct Next State
                if next_state not in transition_probs:
                    transition_probs[next_state] = prob
                else:
                    transition_probs[next_state] += prob

        else:
            candidates = [a for a in list(range(-4, 5, 1))]
            for a in candidates:
                if a == action:
                    prob = self.moveProb
                else:
                    prob = (1 - self.moveProb) / 8

                next_state = self._move(state, a)
                if next_state not in transition_probs:
                        transition_probs[next_state] = prob
                else:
                    transition_probs[next_state] += prob

        return transition_probs

    def rewardFunc(self, agentLoc):
        '''
        Reward
        :param state: Location of Agent State
        :return: Reward Value
        '''
        terminated = agentLoc == self._target_location
        reward = 10 if terminated else 0

        # Penalty for Arriving Slow
        if not terminated:
            reward += self.defaultReward

        if self.attrTable['Pass'][agentLoc] != 1:
            reward = -1

        return reward

    def reset(self, seed=None, origin=12, destination=106, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._agent_location = origin
        self._target_location = destination
        observation = self._get_obs()
        info = self._get_info()
        state = list(observation.values())

        return state, info

    def _move(self, state, action):
        '''

        :param state: 智能体所处的位置
        :param action: 智能体在当前位置所采取的动作
        :return: 智能体的下一个位置
        '''

        row, col = self.state_to_coordinate(state)
        next_row, next_col = row, col

        # 通过行动来转移状态
        if action == -1:
            next_col -= 1
        elif action == 2:
            next_row += 1
        elif action == 1:
            next_col += 1
        elif action == -2:
            next_row -= 1
        elif action == -4:
            next_row -= 1
            next_col += 1
        elif action == 3:
            next_row -= 1
            next_col -= 1
        elif action == -3:
            next_row += 1
            next_col += 1
        elif action == 4:
            next_row += 1
            next_col -= 1
        elif action == 0:
            next_row += 0
            next_col += 0

        # 检测是否超出grid
        if not (0 <= next_row < self.nrow):
            next_row, next_col = row, col
        if not (0 <= next_col < self.ncol):
            next_row, next_col = row, col

        next_state = self.coordinate_to_state(next_row, next_col)

        # 检测下一个状态是否为可用状态
        if self.attrTable['Pass'][next_state] != 1:
            next_state = state

        return next_state

    def step(self, action):
        self._agent_location = self._move(self._agent_location, action)

        # An episode is done if the agent has reached the target
        terminated = self._agent_location == self._target_location

        # Sparse Rewards
        reward = self.rewardFunc(self._agent_location)

        observation = self._get_obs()
        info = self._get_info()
        nextState = list(observation.values())

        return nextState, reward, terminated, info

if __name__ == '__main__':
    os.chdir(r'F:\BaiduNetdiskDownload\共享单车轨迹\共享单车轨迹\01_研究生毕业论文\1_2_DataPreProcess')
    attrTable = pd.read_csv(r'./stateAttrNorm.csv', sep=',')

    env = bicycleGridRiding(attrTable=attrTable)
    state, _ = env.reset()
    print(env.transitFuncArray(state=11, action=-1))
    print(env.transitFunc(state=11, action=-1))
