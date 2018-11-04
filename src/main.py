import gym 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import transform
import numpy as np
from collections import deque
import tensorflow as tf
import random
import src.DQNetwork
import src.HyperParmeters
import src.Memory
import src.Helper


def initial_experience_replay(params:src.HyperParmeters.HyperParameters, env:gym.Env, possible_actions:np.array):
    helper = src.Helper.Helper()
    memory = src.Memory.Memory(max_size = params.memory_size)
    stack_size = 4 # We stack 4 frames

    state = env.reset()
    img_shape = helper.preProcessImage(state).shape

    # Initialize deque with zero-images one array for each image
    stacked_frames  =  deque([np.zeros(img_shape, dtype=np.int) for i in range(stack_size)], maxlen=4)
    for i in range(params.pretrain_length):
        # If it's the first step
        if i == 0:
            state = env.reset()
            
            state, stacked_frames = helper.stack_frames(stacked_frames, state, True, img_shape[0], img_shape[1] )
            
        # Get the next_state, the rewards, done by taking a random action
        choice = random.randint(0,len(possible_actions))-1
        action = possible_actions[choice]
        next_state, reward, done, _ = env.step(choice)
        
        #env.render()
        
        # Stack the frames
        next_state, stacked_frames = helper.stack_frames(stacked_frames, next_state, False, img_shape[0], img_shape[1])
        
        
        # If the episode is finished (we're dead 3x)
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)
            
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            
            # Start a new episode
            state = env.reset()
            
            # Stack the frames
            state, stacked_frames = helper.stack_frames(stacked_frames, state, True, img_shape[0], img_shape[1])
            
        else:
            # Add experience to memory
            memory.add((state, action, reward, next_state, done))
            
            # Our new state is now the next_state
            state = next_state
    return memory



if __name__ == "__main__": 
    env = gym.make('SpaceInvaders-v0')
    init_frame = env.reset()
    # env.render()
    
    # Hyperparameter
    param = src.HyperParmeters.HyperParameters(env.action_space.n)  
    possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    memory = initial_experience_replay(param, env, possible_actions)
    
    
    # Instantiate the DQNetwork
    DQNetwork = src.DQNetwork.DQNetwork(param.state_size, param.action_size, param.learning_rate)    
    writer = tf.summary.FileWriter("/tensorboard/dqn/1")
    tf.summary.scalar("Loss", DQNetwork.loss)
    write_op = tf.summary.merge_all()

    # Reset the graph
    tf.reset_default_graph()

    



