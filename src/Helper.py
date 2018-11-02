import gym 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import transform
import numpy as np
from collections import deque
import tensorflow as tf
import random


class Helper:
        
    def preProcessImage(self, observation):
        img = rgb2gray(observation) 
        img = img[20:-12,4:-12] # crop image
        img = img / 255 # normalize image
        img = transform.rescale(img, 1/1.9)
        return img
    
    def stack_frames(self, stacked_frames, state, is_new_episode, img_shape, stack_size):
        # Preprocess frame
        frame = self.preProcessImage(state)
        
        if is_new_episode:
            # Clear our stacked_frames
            stacked_frames = deque([np.zeros(img_shape, dtype=np.int) for i in range(stack_size)], maxlen=4)
            
            # Because we're in a new episode, copy the same frame 4x
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            stacked_frames.append(frame)
            
            # Stack the frames
            stacked_state = np.stack(stacked_frames, axis=2)
            
        else:
            # Append frame to deque, automatically removes the oldest frame
            stacked_frames.append(frame)

            # Build the stacked state (first dimension specifies different frames)
            stacked_state = np.stack(stacked_frames, axis=2) 
        
        return stacked_state, stacked_frames