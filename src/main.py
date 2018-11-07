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
    return memory, stacked_frames


def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions, sess, DQNetwork):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = random.randint(1,len(possible_actions))-1
        #action = possible_actions[choice]
        
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = choice
        #action = possible_actions[choice]
                
                
    return action, explore_probability


def trainingLoop(params:src.HyperParmeters.HyperParameters, env:gym.Env, possible_actions:np.array,
        DQNetwork:src.DQNetwork.DQNetwork, stacked_frames, memory:src.Memory.Memory):
    
    
    with tf.Session() as sess:
        # Initialize the variables
        helper = src.Helper.Helper()
        sess.run(tf.global_variables_initializer())
        # saver = tf.train.Saver()
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        
        for episode in range(params.total_episodes):
            # Set step to 0
            step = 0
            loss = 0
            
            # Initialize the rewards of the episode
            episode_rewards = []
            rewards_list = []

            
            # Make a new episode and observe the first state
            state = env.reset()
            img_shape = helper.preProcessImage(state).shape
            
            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = helper.stack_frames(stacked_frames, state, True, img_shape[0], img_shape[1])
            
            while step < params.max_steps:
                step += 1
                
                #Increase decay_step
                decay_step +=1
                
                # Predict the action to take and take it
                action, explore_probability = predict_action(params.explore_start, params.explore_stop, params.decay_rate, decay_step, state, possible_actions, sess, DQNetwork)
                
                #Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ = env.step(action)
                
                if params.episode_render:
                    env.render()
                
                # Add the reward to total reward
                episode_rewards.append(reward)
                
                # If the game is finished
                if done:
                    # The episode ends so no next state
                    next_state = np.zeros((86,72), dtype=np.int)
                    
                    next_state, stacked_frames = helper.stack_frames(stacked_frames, next_state, False, img_shape[0], img_shape[1])

                    # Set step = max_steps to end the episode
                    step = params.max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                                'Total reward: {}'.format(total_reward),
                                'Explore P: {:.4f}'.format(explore_probability),
                                'Training Loss {:.4f}'.format(loss))

                    rewards_list.append((episode, total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    memory.add((state, action, reward, next_state, done))

                else:
                    # Stack the frame of the next_state
                    next_state, stacked_frames = helper.stack_frames(stacked_frames, next_state, False, img_shape[0], img_shape[1])
                
                    # Add experience to memory
                    memory.add((state, action, reward, next_state, done))

                    # st+1 is now our current state
                    state = next_state
                    

                ### LEARNING PART            
                # Obtain random mini-batch from memory
                batch = memory.sample(params.batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch]) 
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})
                
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + params.gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                                        feed_dict={DQNetwork.inputs_: states_mb,
                                                DQNetwork.target_Q: targets_mb,
                                                DQNetwork.actions_: actions_mb})

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                    DQNetwork.target_Q: targets_mb,
                                                    DQNetwork.actions_: actions_mb})
                writer.add_summary(summary, episode)
                writer.flush()

            # Save model every 5 episodes
            # if episode % 5 == 0:
            #     save_path = saver.save(sess, "./models/model.ckpt")
            #     print("Model Saved")
    

if __name__ == "__main__": 
    env = gym.make('SpaceInvaders-v0')
    init_frame = env.reset()
    # env.render()
    
    # Hyperparameter
    param = src.HyperParmeters.HyperParameters(env.action_space.n)  
    possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    memory, stacked_frames = initial_experience_replay(param, env, possible_actions)
    
    
    # Instantiate the DQNetwork
    DQNetwork = src.DQNetwork.DQNetwork(param.state_size, param.action_size, param.learning_rate)    
    writer = tf.summary.FileWriter("/tensorboard/dqn/1")
    tf.summary.scalar("Loss", DQNetwork.loss)
    write_op = tf.summary.merge_all()

    # Reset the graph
    tf.reset_default_graph()
    trainingLoop(param, env, possible_actions, DQNetwork, stacked_frames, memory)

