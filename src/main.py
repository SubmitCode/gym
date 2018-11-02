import gym 
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import transform
import numpy as np
from collections import deque
import tensorflow as tf
import random


def preProcessImage(observation):
    img = rgb2gray(observation) 
    img = img[20:-12,4:-12] # crop image
    img = img / 255 # normalize image
    img = transform.rescale(img, 1/1.9)
    return img


if __name__ == "__main__": 
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    env.render()
