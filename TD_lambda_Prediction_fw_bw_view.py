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

# returns Numpy Array of discounted rewards bw two given time_steps
def compute_discounted_reward(discount, reward, start_time_step, end_time_step):
    # compute G(t) for each time step ##############
    # G(t) = Rt + d*G(t+1)
    time_diff = end_time_step - start_time_step
    g = np.zeros(time_diff + 1)
    i = end_time_step
    j = time_diff
    prev_g = 0
    while i >= start_time_step:
        g[j] = reward[i] + discount * prev_g
        prev_g = g[j]
        i = i - 1
        j = j - 1
    return g[0]
#################################################

### TD-Lambda ###
no_of_episodes = 100000
TIME_STEP_LIMIT = 50
discount = 0.9
alpha = 0.02
td_N = 0
state_value = defaultdict(float)
start_time = time()
lambda_ = 0.5
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
            target_state = time_step- td_N-1  # state for which value is updated
            discounted_reward = compute_discounted_reward(discount, reward,target_state,time_step)
            for j in range(td_N+1):
                discounted_reward = discounted_reward +  pow(lambda_ , j) * compute_discounted_reward(discount, reward, j+1, time_step)
            lambda_discounted_reward = (1-lambda_) * discounted_reward
            td_error = (discounted_reward - state_value[state_list[target_state]])
            state_value[state_list[target_state]] = state_value[state_list[target_state]] + alpha * td_error

        ##########################################################
        if isTerminate:
            # computed updates for last td_N+1 states of episode
            # compute G(t) for each time step ##############
            # G(t) = Rt + d*G(t+1)
            g = np.zeros(td_N + 1)
            current_step = time_step
            prev_g = 0
            reward_index = 0
            while current_step >= time_step - td_N:
                g[reward_index] = reward[current_step] + discount * prev_g
                prev_g = g[reward_index]
                current_step = current_step - 1
                reward_index = reward_index + 1
            #################################################

            # calculate v(s) for every state ################
            ts = time_step - td_N
            reward_index = 0
            while ts < time_step + 1:
                # if state_list[i] in state_value:
                # current state has already occured, incremental mean
                td_error = (g[reward_index] - state_value[state_list[ts]])
                new_mean = state_value[state_list[ts]] + alpha * td_error
                state_value[state_list[ts]] = new_mean
                ts = ts + 1
                reward_index = reward_index + 1
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
