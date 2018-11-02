
class HyperParameters:

    def __init__(self, action_space):
        ### MODEL HYPERPARAMETERS
        self.state_size = [94, 76, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels) 
        self.action_size = action_space    # 8 possible actions
        self.learning_rate =  0.00025      # Alpha (aka learning rate)

        ### TRAINING HYPERPARAMETERS
        self.total_episodes = 50            # Total episodes for training
        self.max_steps = 50000              # Max possible steps in an episode
        self.batch_size = 64                # Batch size

        # Exploration parameters for epsilon greedy strategy
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 
        self.decay_rate = 0.00001           # exponential decay rate for exploration prob

        # Q learning hyperparameters
        self.gamma = 0.9                    # Discounting rate

        ### MEMORY HYPERPARAMETERS
        self.pretrain_length = self.batch_size   # Number of experiences stored in the Memory when initialized for the first time
        self.memory_size = 1000000          # Number of experiences the Memory can keep

        ### PREPROCESSING HYPERPARAMETERS
        self.stack_size = 4                 # Number of frames stacked

        ### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
        self.training = False

        ## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
        self.episode_render = False