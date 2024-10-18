import numpy as np
import random


par_mut_rate = 0.005

def tweak(x):

    if random.uniform(0,1) < par_mut_rate:

        rand = random.uniform(-1,1)
        res = x+x*rand

        return res
    
    return x

array_tweak = np.vectorize(tweak)



def select_parent(pop):

    rand_num = random.uniform(0,sum([x.score for x in pop]))
    running_sum = 0

    for agent in pop:
        running_sum+=agent.score

        if running_sum>rand_num:
            return agent.nn



def select_parents(pop):
    parent_0 = select_parent(pop)
    parent_1 = select_parent(pop)

    for _ in range(100):

        if parent_0 != parent_1:

            return (parent_0, parent_1)
        
        parent_1 = select_parent(pop)


    return (parent_0, parent_1)



def crossover(parent0, parent1):

    size = 0
    p_shape = parent0.shape

    for i, n in enumerate(p_shape[1:]):
        size += p_shape[i]*n

    rand = random.randint(0, size)

    ch0w = [] #child0 weights
    ch0b = [] #child0 biases
    ch1w = [] #child1 weights
    ch1b = [] #child1 biases



    for i, n in enumerate(parent0.weights):

        rand -= n.shape[0]*n.shape[1]
 
        if rand<0:
            rand = n.shape[0]*n.shape[1] + rand

            base = int(rand/n.shape[1])
            res = rand%n.shape[1]
            
            if res==0:
                base-=1

            w0 = np.zeros(n.shape)
            w1 = np.zeros(n.shape)
            b0 = np.zeros(parent0.biases[i].shape)
            b1 = np.zeros(parent0.biases[i].shape)


            w0[:base, :] = n[:base, :]
            w0[base, :res] = n[base, :res]
            w0[base, res:] = parent1.weights[i][base, res:] 
            w0[base+1:, :] = parent1.weights[i][base+1:, :]

            w1[:base, :] = parent1.weights[i][:base, :]
            w1[base, :res] = parent1.weights[i][base, :res]
            w1[base, res:] = n[base, res:]
            w1[base+1:, :] = n[base+1:, :]

            b0[:base] = parent0.biases[i][:base]
            b0[base:] = parent1.biases[i][base:]

            b1[:base] = parent1.biases[i][:base]
            b1[base:] = parent0.biases[i][base:]


            ch0w.append(w0)
            ch1w.append(w1)

            ch0b.append(b0)
            ch1b.append(b1)

            ch0w += parent1.weights[i+1:]
            ch1w += parent0.weights[i+1:]

            ch0b += parent1.biases[i+1:]
            ch1b += parent0.biases[i+1:]

            break

        ch0w.append(parent0.weights[i])
        ch0b.append(parent0.biases[i])
        ch1w.append(parent1.weights[i])
        ch1b.append(parent1.biases[i])


    return ((ch0w, ch0b),(ch1w, ch1b))


def cal_size(nn):
    out = 0
    for weights in nn.weights:
        out += np.sum(weights)
    
    return out

def normalize(x, min, max):
    return (x - min)/(max-min)

def denormalize(x, min, max):
    return x * (max - min) + min





