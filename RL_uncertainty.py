from __future__ import division
import numpy as np
from copy import deepcopy
import matplotlib
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy.linalg import norm
import plotting
import gym
from gym.envs.toy_text import discrete
import sys
import tensorflow as tf
from collections import deque
from scipy.spatial.distance import euclidean

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
NONE = 4

class WindyGridworldEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta, winds):
        new_position = np.array(current) + np.array(delta) + np.multiply(1, self.winds_convert_dict[winds[tuple(current)]])
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = tuple(new_position) == self.terminal_state
        if is_done:
            return [(1.0, new_state, 10000000, is_done)]
        if delta == [0,0]:
            return [(1.0, new_state, -euclidean(np.unravel_index(new_state,self.shape),self.terminal_state), is_done)]
        return [(1.0, new_state, -5*euclidean(np.unravel_index(new_state,self.shape),self.terminal_state), is_done)]

    def __init__(self):
        self.shape = (50, 50)

        nS = np.prod(self.shape)
        nA = 5

        # Wind strength
        winds = add_square_current((10,10),5,self.shape)
        winds += add_square_current((10,10),4,self.shape)
        winds += add_square_current((10,10),3,self.shape)
        winds += add_square_current((10,10),2,self.shape)
        
        for i in range(1,6):
            winds += add_square_current((30,30),i,self.shape)
        self.winds = winds
        self.winds_convert_dict = {0:[0,0],1:[0,1],2:[1,0],3:[0,-1],4:[-1,0]}
        self.start_state = (10,3)
        self.terminal_state = (30,40)
        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][UP] = self._calculate_transition_prob(position, [-1, 0], winds)
            P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1], winds)
            P[s][DOWN] = self._calculate_transition_prob(position, [1, 0], winds)
            P[s][LEFT] = self._calculate_transition_prob(position, [0, -1], winds)
            P[s][NONE] = self._calculate_transition_prob(position, [0, 0], winds)

#       Set start point
        isd = np.zeros(nS)
        isd[np.ravel_multi_index(self.start_state, self.shape)] = 1.0

        super(WindyGridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == self.terminal_state:
                output = " T "
            elif self.winds[tuple(position)] != 0:
                output = " "+str(int(self.winds[tuple(position)]))+" "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")
        
    def show_img(self):
        world = deepcopy(self.winds)
        world[np.unravel_index(self.s,self.shape)] = 7
        plt.imshow(world)

def limit_coordinates(coord,world):
    coord[0] = min(coord[0], np.shape(world)[0] - 1)
    coord[0] = max(coord[0], 0)
    coord[1] = min(coord[1], np.shape(world)[1] - 1)
    coord[1] = max(coord[1], 0)
    return coord

def add_square_current(center,radius,world_shape,direction='cw'):
    world = np.zeros(world_shape)
    vertical_boundary = [center[0]-radius, center[0]+radius]
    vertical_boundary = limit_coordinates(vertical_boundary,world)
    horizontal_boundary = [center[1]-radius, center[1]+radius]
    horizontal_boundary = limit_coordinates(horizontal_boundary,world)
    vertical_index = np.linspace(vertical_boundary[0],vertical_boundary[1],vertical_boundary[1]+1-vertical_boundary[0])
    horizontal_index = np.linspace(horizontal_boundary[0],horizontal_boundary[1],horizontal_boundary[1]+1-horizontal_boundary[0])
    xx, yy = np.meshgrid(vertical_index,horizontal_index)
    x_range = np.array([np.amin(xx),np.amax(xx)]).astype(int)
    y_range = np.array([np.amin(yy),np.amax(yy)]).astype(int)
    xxyy = zip(xx,yy)
    coord = []
    for i,j in xxyy:
        coord += zip(i,j)
    coord = np.array([np.array(i).astype(int) for i in coord])
    boundary_index = []
    for i,j in coord:
        if i in x_range or j in y_range:
            boundary_index += [[i,j]]
#   1:right, 2:down, 3:left, 4:up, 0:no current
    if direction == 'cw':
        world[y_range[0]:y_range[1]+1,x_range[0]] = 4
        world[y_range[0]:y_range[1],x_range[1]] = 2
        world[y_range[0],x_range[0]:x_range[1]] = 1
        world[y_range[1],x_range[0]+1:x_range[1]+1] = 3
    elif direction == 'ccw':
        world[y_range[0]:y_range[1]+1,x_range[0]] = 2
        world[y_range[0]:y_range[1],x_range[1]] = 4
        world[y_range[0],x_range[0]:x_range[1]] = 3
        world[y_range[1],x_range[0]+1:x_range[1]+1] = 1
    else:
        raise AttributeError("Direction input is not correct, please use 'cw' or 'ccw'")
    return world

def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def get_optimal_path(Q,env):
    env.reset()
    start_state = env.start_state
    terminal_state = env.terminal_state
    state = np.ravel_multi_index(start_state,env.shape)
    path = [start_state]
    value = 0
    action = []
    while 1:
        next_action = np.argmax(Q[state])
        next_state,reward,done,_ = env.step(next_action)
        path += [np.unravel_index(next_state,env.shape)]
        value += reward
        action += [next_action]
        if done:
            return path, action, value
            break
        state = next_state

if __name__ == "__main__":


	env = WindyGridworldEnv()

	#These lines establish the feed-forward part of the network used to choose actions
	inputs1 = tf.placeholder(shape=[None,1],dtype=tf.float32)
	W1 = tf.Variable(tf.random_uniform([1,10],0,0.01))
	B1 = tf.Variable(tf.zeros([10]))
	L1 = tf.add(tf.matmul(inputs1,W1),B1)
	L1 = tf.nn.sigmoid(L1)
	W2 = tf.Variable(tf.random_uniform([10,5],0,0.01))
	B2 = tf.Variable(tf.zeros([5]))
	L2 = tf.add(tf.matmul(L1,W2),B2)
	# L2 = tf.nn.sigmoid(L2)
	Qout = L2
	predict = tf.argmax(Qout,1)

	#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
	nextQ = tf.placeholder(shape=[1,5],dtype=tf.float32)
	loss = tf.reduce_sum(tf.square(nextQ - Qout))
	trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
	updateModel = trainer.minimize(loss)

	init = tf.global_variables_initializer()
	discount_factor = 0.7
	e = 0.9
	_lambda = 0.7
	num_episodes = 2000
	stats = plotting.EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards=np.zeros(num_episodes))
	with tf.Session() as sess:
	    sess.run(init)
	    for i in range(num_episodes):
	        #Reset
	        E = defaultdict(lambda:np.zeros(env.action_space.n))
	        state = env.reset()
	        done = False
	        state_list = deque(maxlen=50)
	        action_list = deque(maxlen=50)
	        action = sess.run(predict,feed_dict={inputs1:[[state]]})
	        if np.random.rand(1) < e:
	            action[0] = env.action_space.sample()
	                
	        for t in itertools.count():
	            Q = sess.run(Qout,feed_dict={inputs1:[[state]]})
	            
	            next_state,reward,done,_ = env.step(action[0])
	            next_action,Q_next = sess.run([predict,Qout],feed_dict={inputs1:[[next_state]]})
	            print "\rEpi:%i,t:%i,state:%s,action:%i,Q:%s" %(i,t,str(np.unravel_index(state,env.shape)),action[0],str(Q)),
	#             sys.stdout.flush()
	            if np.random.rand(1) < e:
	                next_action[0] = np.random.randint(0,5)
	                
	            delta = reward + discount_factor*Q_next[0][next_action[0]] - Q[0][action[0]]
	            E[state][action] += 1
	            state_list.append(state)
	            action_list.append(action)
	            
	            for s,a in zip(state_list,action_list):
	                Q_sa = sess.run(Qout,feed_dict={inputs1:[[s]]})
	                targetQ = Q_sa + delta*E[s][a[0]]
	                sess.run(updateModel,feed_dict={inputs1:[[s]],nextQ:targetQ})
	                E[s][a[0]] *= _lambda * discount_factor
	                
	            stats.episode_rewards[i] += reward
	            stats.episode_lengths[i] = t
	            
	            state = next_state
	            action = next_action
	            
	            if done == True:
	                #Reduce chance of random action as we train the model.
	                e = 1./((i/50) + 10)
	                break
	            if t>=10000:
	                break
	print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

	matplotlib.style.use('ggplot')
	plotting.plot_episode_stats(stats)

	opt_path,action,value = get_optimal_path(Q,env)