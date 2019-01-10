import numpy as np
from RLalgs.utils import epsilon_greedy
import random

def SARSA(env, num_episodes, gamma, lr, e):
    """
    Implement the SARSA algorithm following epsilon-greedy exploration.

    Inputs:
    env: OpenAI Gym environment 
            env.P: dictionary
                    P[state][action] are tuples of tuples tuples with (probability, nextstate, reward, terminal)
                    probability: float
                    nextstate: int
                    reward: float
                    terminal: boolean
            env.nS: int
                    number of states
            env.nA: int
                    number of actions
    num_episodes: int
            Number of episodes of training
    gamma: float
            Discount factor.
    lr: float
            Learning rate.
    e: float
            Epsilon value used in the epsilon-greedy method.

    Outputs:
    Q: numpy.ndarray
            State-action values
    """
    
    Q = np.zeros((env.nS, env.nA))
    
    #TIPS: Call function epsilon_greedy without setting the seed
    #      Choose the first state of each episode randomly for exploration.
    ############################
    # YOUR CODE STARTS HERE
    for i in range(num_episodes):
        state = np.random.randint(0, env.nS)
        action = epsilon_greedy(Q[state], e)
        while len(env.P[state][0]) != 1:
            index = np.random.randint(0, 3)
            R = env.P[state][action][index][2]
            S = env.P[state][action][index][1]
            A = epsilon_greedy(Q[S], e)
            Q[state, action] = Q[state, action] + lr * (R + gamma * Q[S, A] - Q[state, action])
            state = S
            action = A
    # YOUR CODE ENDS HERE
    ############################

    return Q