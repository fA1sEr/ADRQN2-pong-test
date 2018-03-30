import itertools as it
import os
from time import time, sleep
import numpy as np
import skimage.color
import skimage.transform
import tensorflow as tf
from tqdm import trange
from vizdoom import *
from Agent import Agent
from GameSimulator import GameSimulator

# to choose gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

FRAME_REPEAT = 4 # How many frames 1 action should be repeated
UPDATE_FREQUENCY = 4 # How many actions should be taken between each network update
COPY_FREQUENCY = 1000

RESOLUTION = (80, 45, 4) # Resolution
BATCH_SIZE = 32 # Batch size for experience replay
LEARNING_RATE = 0.001 # Learning rate of model
GAMMA = 0.99 # Discount factor

MEMORY_CAP = 1000 # Amount of samples to store in memory

EPSILON_MAX = 1 # Max exploration rate
EPSILON_MIN = 0.05 # Min exploration rate
EPSILON_DECAY_STEPS = 2e5 # How many steps to decay from max exploration to min exploration

RANDOM_WANDER_STEPS = 200000 # How many steps to be sampled randomly before training starts

TRACE_LENGTH = 8 # How many traces are used for network updates
HIDDEN_SIZE = 768 # Size of the third convolutional layer when flattened

EPOCHS = 20000000 # Epochs for training (1 epoch = 200 training Games and 10 test episodes)
GAMES_PER_EPOCH = 200 # How actions to be taken per epoch
EPISODES_TO_TEST = 5 # How many test episodes to be run per epoch for logging performance
EPISODE_TO_WATCH = 10 # How many episodes to watch after training is complete

TAU = 0.99 # How much the target network should be updated towards the online network at each update

LOAD_MODEL = True # Load a saved model?
SAVE_MODEL = True # Save a model while training?
SKIP_LEARNING = False # Skip training completely and just watch?

max_model_savefile = "train_data/max_model/max_model.ckpt"
model_savefile = "../ADRQN2-pong0.5/train_data/max_model/max_model.ckpt" # Name and path of the model
reward_savefile = "train_data/Rewards.txt"

##########################################

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

def saveScore(score):
    my_file = open(reward_savefile, 'a')  # Name and path of the reward text file
    my_file.write("%s\n" % score)
    my_file.close()

###########################################

game = GameSimulator()
game.initialize()

ACTION_COUNT = game.get_action_size()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)

SESSION = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if LOAD_MODEL:
    EPSILON_MAX = 0.25 # restart after 20+ epoch

agent = Agent(memory_cap = MEMORY_CAP, batch_size = BATCH_SIZE, resolution = RESOLUTION, action_count = ACTION_COUNT,
            session = SESSION, lr = LEARNING_RATE, gamma = GAMMA, epsilon_min = EPSILON_MIN, trace_length=TRACE_LENGTH,
            epsilon_decay_steps = EPSILON_DECAY_STEPS, epsilon_max=EPSILON_MAX, hidden_size=HIDDEN_SIZE)

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, TAU)

print("Loading model from: ", model_savefile)
saver.restore(SESSION, model_savefile)

##########################################
print("\nTesting...")

test_scores = []

for test_step in range(EPISODES_TO_TEST):
    game.reset()
    agent.reset_cell_state()
    while not game.is_terminared():
        state = game.get_state()
        action = agent.act(state, train=False)
        game.make_action(action)
    now_score = game.get_total_reward()
    saveScore(now_score)
    test_scores.append(now_score)

test_scores = np.array(test_scores)
my_file = open(reward_savefile, 'a')  # Name and path of the reward text file
my_file.write("%.1f (±%.1f)  min:%.1f  max:%.1f\n" % (test_scores.mean(), test_scores.std(), test_scores.min(), test_scores.max()))
my_file.close()
