""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import _pickle as pickle
import retro
import time
import json

#### DOCUMENTATION VIA NUMPY ####
# np.exp() => exponentiel calculation of each input array 's elements
# particularly useful in neural networks to calculate the gradient of the error
# .ravel() => flatten a table N-D in 1-D

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factorpong for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
render = True

total_reward_sum = []
total_running_reward = []

# model initialization
# D = 80 * 80  # input dimensionality: 80x80 grid
D = 120 * 80  # input dimensionality: 120x120 grid # First modification
if resume:
    model = pickle.load(open('save.p', 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization -> Initialisation pseudo-random neuron weight
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k, v in iter(model.items())}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in iter(model.items())}  # rmsprop memory


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def prepro(I): # frame 240x160x3 uint8 frame
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """ # -> Celeste : frame size 240x160x3 -> crop 120x80
    # I = I[35:195]  # crop -> pas de crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

# Computing h (hidden state) with W1 (first state)
# to compute logp with the hidden state and the last state (W2).
def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking any actions, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu : calculation how to modify W1 and W2 in relation to the reward/penalty
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}

def writeInFile(total_reward_sum, total_running_reward):
    """Write some rewards data in a file"""
    data = {}
    data['records'] = []

    for i in range(len(total_reward_sum)):
        data['records'].append({
            'rewardSum': total_reward_sum[i],
            'runningReward': total_running_reward[i]
        })

    with open('runRewardRecords.json', 'w') as outfile:
        json.dump(data, outfile)




env = retro.make(game="Celeste-GBA", use_restricted_actions=retro.Actions.MULTI_DISCRETE)
observation = env.reset()
prev_x = None  # used in computing the difference frame
xs, hs1, hs2, hs3, dlogps1, dlogps2, dlogps3, drs = [], [], [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 0
actionIndex = 0.0
t0 = time.time()
while True:
    if render: env.render()

    actionIndex += 0.01

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob1, h1 = policy_forward(x)
    aprob2, h2 = policy_forward(x)
    aprob3, h3 = policy_forward(x)

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs1.append(h1)  # hidden state
    hs2.append(h2)  # hidden state
    hs3.append(h3)  # hidden state
    
    # Action choice with ranges

    # elem1 = 0
    # if aprob1 <= 0.49:
    #     elem1 = 0
    # elif aprob1 <= 0.50:
    #     elem1 = 1
    # else:
    #     elem1 = 2

        
    # elem2 = 0
    # if aprob2 <= 0.49:
    #     elem2 = 0
    # elif aprob2 <= 0.50:
    #     elem2 = 1
    # else:
    #     elem2 = 2

    # elem3 = 0
    # if aprob3 <= 0.485:
    #     elem3 = 0
    # elif aprob3 <= 0.495:
    #     elem3 = 1
    # elif aprob3 <= 0.505:
    #     elem3 = 2
    # else:
    #     elem3 = 4


    # Action choice
    elem1 = 1 if np.random.uniform() < aprob1 else 2
    elem2 = 1 if np.random.uniform() < aprob2 else 2
    elem31 = 1 if np.random.uniform() < aprob3 else 0
    elem32 = 2 if np.random.uniform() < aprob3 else 0

    if elem31 != 0 and elem32 != 0:
        elem3 = 4
    elif elem31 == 1:
        elem3 = 1
    elif elem32 == 2:
        elem3 = 2 
    else:
        elem3 = 0

    # Fake labe http://cs231n.github.io/neural-networks-2/#losses
    dlogps1.append(elem1 - aprob1)
    dlogps2.append(elem2 - aprob2)
    dlogps3.append(elem3 - aprob3)

    # Action
    a = [elem1, elem2, elem3, 0]

    # step the environment and get new measurements
    observation, reward, done, info = env.step(a)

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)

    if done:  # an episode finished
        t1 = time.time()             
        if t1-t0 > 1:        
            reward_sum += 0.1*((t1-t0)//2)    

        episode_number += 1
        

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph1 = np.vstack(hs1)
        eph2 = np.vstack(hs2)
        eph3 = np.vstack(hs3)
        epdlogp1 = np.vstack(dlogps1)
        epdlogp2 = np.vstack(dlogps2)
        epdlogp3 = np.vstack(dlogps3)
        epr = np.vstack(drs)
        xs, hs1, hs2, hs3, dlogps1, dlogps2, dlogps3, drs = [], [], [], [], [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards pong to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp1 *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp2 *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp3 *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad1 = policy_backward(eph1, epdlogp1)
        grad2 = policy_backward(eph2, epdlogp2)
        grad3 = policy_backward(eph3, epdlogp3)
        for k in model: grad_buffer[k] += grad1[k]  # accumulate grad over batch
        for k in model: grad_buffer[k] += grad2[k]  # accumulate grad over batch
        for k in model: grad_buffer[k] += grad3[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in iter(model.items()):
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        total_reward_sum.append(reward_sum)
        total_running_reward.append(running_reward)

        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: game finished, reward: %f' % (episode_number, reward))
        writeInFile(total_reward_sum, total_running_reward)
