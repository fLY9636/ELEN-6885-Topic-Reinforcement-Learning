# This file contains all the constants in the project.

# System Setting
USE_GPU = False                  # To use GPU, set True
VERBOSE = False                  # print out result

# File saving folder
LOG_FILE = 'runing_logs'        # name of log folder
CHECKPOINT_DIR = 'saved_model'  # name of model saving folder

# RMSProp Parameters
RMSP_ALPHA = 0.99               # decay parameter 
RMSP_EPSILON = 0.1              # epsilon parameter

# Learning Rate Initialization
LR_ALPHA_LOW = 1e-4             # log_uniform: low limit of learning rate 
LR_ALPHA_HIGH = 1e-2            # log_uniform: high limit of learning rate
LR_ALPHA_LOG_RATE = 0.4226      # log_uniform interpolate rate for learning rate (around 7 * 10^-4)

# Environment Parameters
ACTION_SIZE = 4
HISTORY_LENGTH = 4
GAMMA = 0.99                    # discount factor for rewards
REWARD_MOVE = -0.01
REWARD_COLLIDE = -0.1
REWARD_TERMINAL = 10
ENTROPY_BETA = 0.01             # entropy parameter used in loss function

# A3C Threads parameter
MAX_TIME_STEP = 10.0 * 10**6    # 10 million frames
NUM_THREADS = 4                # number of parallel threads
LOCAL_T_MAX = 5                 # number of local accumulated steps
GRAD_NORM_CLIP = 40.0           # gradient norm clipping

# Evaluation Parameter
NUM_EVAL_EPISODES = 1         # number of episodes for evaluation

# DQN parameter (used in the thread)
MEMORY_SIZE = 1
DQN_BATCH_SIZE = 1
DQN_REPLACE_TARGET_ITER = 200
REWARD_DECAY = 0.9

# Scenes and tasks
TASK_TYPE = 'navigation'        # scope name
TASK_LIST = {
  'bathroom_02'     : ['26', '69'],
  'kitchen_02'      : ['134', '329']
  #'bathroom_02'    : ['26', '37', '43', '53', '69'],
  #'bedroom_04'     : ['134', '264', '320', '384', '387'],
  #'kitchen_02'     : ['90', '136', '157', '207', '329'],
  #'living_room_08' : ['92', '135', '193', '228', '254']
}                               # keys are scene names, and values are ids of location of navigation targets

# DQN parameter
EPSILON = 0.9











