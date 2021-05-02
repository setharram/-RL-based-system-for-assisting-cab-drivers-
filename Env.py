# Import routines

import numpy as np
import math
import random

from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 2),
         (1, 3), (1, 4), (2, 0), (2, 1), (2, 3), (2, 4), (3, 0),
         (3, 1), (3, 2), (3, 4), (4, 0), (4, 1), (4, 2), (4, 3)]"""
        self.action_space = [(0, 0)]  + list(permutations([i for i in range(m)], 2))
        
        ''' The state space is defined by the driver’s current location along \
        with the time components (hour-of-the-day and the day-of-the-week). \
        A state is defined by three variables: s = Xi Tj Dk 
        where i = 0 ... m − 1; j = 0 ... . t − 1; k = 0 ... . . d − 1'''
        self.state_space = [[a, b, c] \
                            for a in range(m) for b in range(t) for c in range(d)]
        
        self.state_init = random.choice(self.state_space)
        
        self.action_size = len(self.action_space)
        self.state_size = (m + t + d)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN.\
        This method converts a given state into a vector format"""
        
        # define state_encode vector
        state_encode = [0] * (m + t + d)
        
        # encode location
        state_encode[state[0]] = 1
        
        # encode hour of the day
        state_encode[m + state[1]] = 1
        
        # encode day of the week
        state_encode[m + t + state[2]] = 1
        return state_encode

    # function for architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN.\
        This method converts a given state-action pair into a vector format."""
        state_encode = [0]* (m+t+d+m+m)
        
        # encode location
        state_encode[state[0]] = 1
        
        # encode hour of the day
        state_encode[m + state[1]] = 1
        
        # encode day of the week
        state_encode[m + t + state[2]] = 1
        
        if (action[0] != 0):
            state_encode[m+t+d+action[0]] = 1
        if (action[1] != 0):
            state_encode[m+t+d+m+action[1]] = 1

        return state_encode

    ## Getting number of requests
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
       
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)
        
        if requests >15:
            requests =15

        possible_actions_idx = random.sample(range(1, (m-1)*m +1), requests)+[0]
        # (0,0) is not considered as customer request, drivers action for rejecting the job
        actions = [self.action_space[i] for i in possible_actions_idx]

        return possible_actions_idx,actions  
    
    def formatsum_time_day(self,start_time,start_day,duration):
        duration = int(duration)
        if (start_time + duration) < 24:
            # week of day remains same update time
            drop_time = start_time + duration
            drop_day = start_day
        else:
            # day is crossing more than 24,
            # convert into appropriate format
            drop_time = (start_time + duration)%24
            # update day
            day = (start_time + duration)//24
            # should be in range between (0-6)
            drop_day = (start_day+day)%7
        return drop_time,drop_day
              
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        cur_loc = state[0]
        pickup_loc = action[0]
        drop_loc = action[1]
        
        cur_time = state[1]
        cur_day  = state[2]
        
        # time duration
        total_dur   = 0
        pickup_dur  = 0    # to go from current  location to pickup location
        wait_dur    = 0    # in case driver chooses to refuse all requests
        ride_dur    = 0    # from Pick-up to drop
        
        if (drop_loc == pickup_loc):
            # driver rejected a(0,0) no ride action, move time component by 1 hr
            wait_dur = 1
            next_loc = cur_loc
        else:
            if (cur_loc != pickup_loc):
                # driver as to go to pickup location
                pickup_dur = Time_matrix[cur_loc][pickup_loc][cur_time][cur_day]
                pickup_time,pickup_day = self.formatsum_time_day(cur_time,cur_day,pickup_dur)       
                ride_dur = Time_matrix[pickup_loc][drop_loc][pickup_time][pickup_day]
            else:
                # driver is already in pickup location
                ride_dur = Time_matrix[pickup_loc][drop_loc][cur_time][cur_day]
            next_loc = drop_loc
            
        total_dur = (wait_dur+pickup_dur+ride_dur)
        next_time,next_day = self.formatsum_time_day(cur_time,cur_day,total_dur)
        
        next_state = [next_loc,next_time,next_day]
        duration = [wait_dur,pickup_dur,ride_dur]
        #print(f"S:{state},A:{action},D:{duration},NS:{next_state}")
        return next_state,duration
    
    def reward_func(self, duration):
        """Takes in wait, transit and ride duration and returns the reward"""
        wait_dur = duration[0]
        tran_dur = duration[1]
        ride_dur = duration[2]
      
        reward = (R*ride_dur)-(C*(ride_dur+wait_dur+tran_dur))
        return reward
    
    def step(self,state,action,Time_matrix):
        
        next_state, duration = self.next_state_func(state, action, Time_matrix)
        reward = self.reward_func(duration)
        total_time = np.sum(duration)
        #print(f"CS:{state},A:{action},NS:{next_state},R:{reward},T:{total_time}")
        return next_state,reward,total_time

    def reset(self):
        return self.action_space, self.state_space, self.state_init
