from env_blackjack import BlackjackEnv
import numpy as np
import matplotlib
import plotting
from collections import defaultdict
from time import time
matplotlib.style.use('ggplot')

bj = BlackjackEnv()

def policy(state):
    score, dealer_score, usable_ace = state
    if score >= 19:
        return 0
    else:
        return 1

### Time-Difference- N ###

no_of_episodes = 100000
TIME_STEP_LIMIT = 50
discount = 0.9
alpha = 0.9
td_N = 0
state_value = defaultdict(float)
start_time = time()
for i in range(no_of_episodes):
    print("#EPISODE"+str(i))
    state_list = [TIME_STEP_LIMIT]
    state_list[0] = bj.reset()
    reward = np.zeros(TIME_STEP_LIMIT, dtype= int)
    action = np.zeros(TIME_STEP_LIMIT, dtype= int)

    for time_step in range(TIME_STEP_LIMIT):
        # SAMPLE STATE, REWARD FROM ENV ##########################
        action[time_step] = policy(state_list[time_step])
        state_, reward[time_step], isTerminate, _ = bj.step(action=action[time_step])
        print("State:" + str(state_list[time_step]) + " Action:" + str(action[time_step]) + " REWARD:" + str(reward[time_step]))
        ##########################################################
        if time_step > td_N:
            # state for which value is updated
            target_state = time_step- td_N-1
            discounted_reward = reward[target_state]
            for j in range(td_N+1):
                discounted_reward = discounted_reward + pow(discount, j + 1) * reward[j + 1]
            td_error = (discounted_reward - state_value[state_list[target_state]])
            new_mean = state_value[state_list[target_state]] + alpha * td_error
            state_value[state_list[target_state]] = new_mean

        ##########################################################
        if isTerminate:
            # computed updates for last td_N+1 states of episode

            # compute G(t) for each time step ##############
            # G(t) = Rt + d*G(t+1)
            g = np.zeros(td_N + 1)
            current_step = time_step
            prev_g = 0
            reward_index = 0
            while current_step >= time_step-td_N:
                g[reward_index] = reward[current_step] + discount * prev_g
                prev_g = g[reward_index]
                current_step = current_step - 1
                reward_index = reward_index+1
            #################################################

            # calculate v(s) for every state ################
            ts = time_step-td_N
            reward_index=0
            while ts < time_step + 1:
                # if state_list[i] in state_value:
                # current state has already occured, incremental mean
                mc_error = (g[reward_index] - state_value[state_list[ts]])
                new_mean = state_value[state_list[ts]] + alpha * mc_error
                state_value[state_list[ts]] = new_mean
                ts = ts+1
                reward_index = reward_index+1
            break
        else:
            state_list.append(state_)
end_time = time()
for state in state_value:
    print("State: " + str(state) + " Value: " + str(state_value[state]))

print("No of States Explored: ", len(state_value))
print("ELAPSED TIME: "+ str(end_time-start_time))
V_10k = state_value
plotting.plot_value_function(V_10k, title="100,000 Steps")
