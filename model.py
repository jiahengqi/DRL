from keras.models import Model
from keras.layers import Input,Dense,Lambda
from keras.optimizers import Adam
from keras import backend as K
import gym
import tensorflow as tf
import numpy as np
from utils import *

from keras.backend.tensorflow_backend import set_session

class dueling_Double_DQN:
    def __init__(self, env, hidden_units=20, maxlen=10000, batch_size=32, 
                 explore_init=0.5, explore_end=0.01, explore_steps=100000,
                 update_fre=20, gamma=0.99, train_times=1, version='double'):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        set_session(tf.Session(config=config))

        self.env=env
        self.batch_size=batch_size
        self.explore=explore_init
        self.explore_init=explore_init
        self.explore_end=explore_end
        self.explore_steps=explore_steps
        self.update_fre=update_fre
        self.gamma=gamma
        self.train_times=train_times
        self.version=version
        
        self.model=self.create_model(hidden_units)
        if self.version.find('nips')<0:
            self.target_model=self.create_model(hidden_units)
        self.memory=Memory(maxlen)
        self.time_stamp=0
        
    def create_model(self, hidden_units):
        x=Input(self.env.observation_space.shape)
        h=Dense(hidden_units, activation='tanh')(x)
        if self.version.find('duel')>-1:
            a=Dense(self.env.action_space.n,)(h)
            v=Dense(1,)(h)
            z=Lambda(lambda a:a[0]+a[1]-K.mean(a[1],keepdims=True))([v,a])
        else:
            z=Dense(self.env.action_space.n,)(h)
        model=Model(inputs=x, outputs=z)
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def update(self):
        self.time_stamp+=1
        if len(self.memory.buffer)>=self.batch_size:
            for _ in range(self.train_times):
                #data=random.sample(self.memory,self.batch_size)
                data=self.memory.sample(self.batch_size)
                s=[d[0] for d in data]
                a=[d[1] for d in data]
                r=[d[2] for d in data]
                s_=[d[3] for d in data]
                done=[d[4] for d in data]
                q1=self.model.predict(np.array(s))
                if self.version.find('nips')>-1:
                    q2=self.model.predict(np.array(s_))
                else:
                    q2=self.target_model.predict(np.array(s_))
                for i in range(self.batch_size):
                    if done[i]:
                        q1[i,a[i]]=r[i]
                    else:
                        if self.version.find('double')>-1:
                            q1[i,a[i]]=r[i]+self.gamma*q2[i, np.argmax(q1[i])]
                        else:
                            q1[i,a[i]]=r[i]+self.gamma*np.max(q2[i])
                self.model.train_on_batch(np.array(s),q1)
        if self.version.find('nips')<0 and self.time_stamp%self.update_fre==0:
            self.target_model.set_weights(self.model.get_weights())
        
    def store(self, s, a, r, s_, done):
        #self.memory.append([s,a,r,s_,done])
        self.memory.add((s, a, r, s_, done))
        
    def get_action(self, s, flag=True):
        if flag:
            if self.explore>self.explore_end:
                self.explore-=(self.explore_init-self.explore_end)/self.explore_steps
            if np.random.rand()<self.explore:
                return self.env.action_space.sample()
        return np.argmax(self.model.predict(s[np.newaxis,:],))