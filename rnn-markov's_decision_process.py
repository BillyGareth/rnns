import numpy as np

# Reward(R) matrix
R = np.matrix([[-1, -1, -1, -1, 0, -1],
               [-1, -1, -1, 0, -1, 100],
               [-1, -1, -1, 0, -1, -1],
               [-1, 0, 0, -1, 0, -1],
               [-1, 0, 0, -1, -1, 100],
               [-1, 0, -1, -1, 0, 100]])

# Q matrix
Q = np.matrix(np.zeros([6, 6]))

# Gamma(learning parameter)
gamma = 0.8

# Initial state (chosen randomly)
Initial_state = 1

# function returns all available actions in the state
def available_actions(state) :
    current_state_row = R[state, ]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

# getting available actions in the current state
available_act = available_actions(Initial_state)

# function chooses which action to be performed within the reach
# out of the available actions
def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act, 1))
    return next_action

# sample next action to be performed
action = sample_next_action(available_act)

# this function updates the Q matrix according to the path selected
def update(current_state, action, gamma):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1]

    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size=1))

    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]

    # Q learning formula
    Q[current_state, action] = R[current_state, action]*gamma * max_value

# update Q matrix
update(Initial_state, action, gamma)

# training with iterations
for i in range(10000):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(current_state)
    action = sample_next_action(available_act)
    update(current_state, action, gamma)

# normalize the trained Q matrix
print('trained Q matrix')
print(Q/np.max(Q)*100)


# training phase
current_state = 1
steps = [current_state]

while current_state != 5:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))

    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index

# Print selected sequence of steps
print('selected paths"')
print(steps)
