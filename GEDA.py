import numpy as np
from time import *


from Problem import DCMaTS
from Problem import GMOTS
from Problem import MCov
from Problem import MCov_Init
from Problem import MCut
from Problem import MCut_Init


def SOI_L(values, m):
    if values.ndim == 1: return sum(values) / m
    else:
        soi = 0
        for j in range(m):
            soi += np.min(values[:, j])     
    return soi / m


def SOI_T(values, m):
    z = np.zeros(m)
    if values.ndim == 1: return np.max(values - z) / m
    else:
        soi_list = []
        for j in range(m):
            soi_list.append(np.min(values[:, j]) - z[j])
        soi = max(soi_list)
        return soi / m


def Greedy(values, k, m, scalar):
    
    s = np.size(values, 0)

    soi_init = []
    for i in range(s):
        if scalar == 1: soi_init.append(SOI_L(values[i], m))
        if scalar == 2: soi_init.append(SOI_T(values[i], m))
    cbest_soi = min(soi_init)
    cbest_idx = [soi_init.index(cbest_soi)]

    for l in range(1, k):
        delta = [-1] * s
        for i in range(s):
            if i in cbest_idx: continue
            cbest_idx.append(i)
            values_cidx = np.array(values[cbest_idx])
            if scalar == 1: delta[i] = cbest_soi - SOI_L(values_cidx, m)
            if scalar == 2: delta[i] = cbest_soi - SOI_T(values_cidx, m)
            del cbest_idx[-1]
        delta = np.array(delta)
        delta_max = np.max(delta)
        delta_max_index = np.where(delta == delta_max)[0][0]
        cbest_idx.append(delta_max_index)
        cbest_soi -= delta_max
        
    return cbest_idx, cbest_soi


if __name__ == '__main__':
    
    problem = 4 # 1: DC-MaTS; 2: GMOTS; 3: MCut; 4: MCuv
    scalar = 1 # scalarization method 1: linear; 2: Tchebycheff 
    
    if problem == 1 or problem == 2:
        pins = 1 # problem instance
        k = 5 # number of solutions
        m = 50 # number of objectives
        n = 75 # variable dimension
        xl = 0 # lower bound of variable
        xu = 1 # upper bound of variable
        ps = 10*m # population size
        ic = 10*m # iteration count 
        
    if problem == 3:
        k = 5
        m = 50
        A, n = MCut_Init(m)
        xl = 0 
        xu = 2 # binary encoding
        ps = 4*m
        ic = 4*m
        
    if problem == 4:
        k = 5
        m = 50
        n = 20 # budget
        values, setlist, num = MCov_Init(m)
        xl = 0
        xu = num # real encoding
        ps = 4*m
        ic = 4*m
      
    runtime = time()
    
    # initialize population
    if problem == 1 or problem == 2: P = np.random.uniform(xl, xu, (ps, n))
    if problem == 3: P = np.random.randint(xl, xu, size=(ps, n))
    if problem == 4: P = np.random.randint(xl, xu, size=(ps, n))
    
    # objective values of population
    O = []
    for i in range(ps):
        if problem == 1: O.append(DCMaTS(P[i], m, n, pins))
        if problem == 2: O.append(GMOTS(P[i], m, n, pins))
        if problem == 3: O.append(MCut(A, P[i], m, n))
        if problem == 4: O.append(MCov(P[i], values, setlist))
    O = np.array(O)
    
    # initialize current best solution set
    cbest_idx, cbest_soi = Greedy(O, k, m, scalar)
    cbest = P[cbest_idx]
    cbest_o = O[cbest_idx]
    
    # initialize global best solution set
    gbest = np.copy(cbest)
    gbest_o = np.copy(cbest_o)
    gbest_soi = np.copy(cbest_soi) 

    for g in range(ic):
        
        for i in range(ps):
            if i in cbest_idx: continue          
            for j in range(n):
                idx = np.random.randint(0, k) # sample a solution from gbest
                if np.random.random() < 1 / 2: # uniform crossover
                    P[i, j] = gbest[idx, j] 
                if np.random.random() < 1 / n: # modified bit-wise mutation
                    if problem == 1 or problem == 2: P[i, j] = np.random.uniform(xl, xu) 
                    if problem == 3: P[i, j] = 1 - P[i, j]
                    if problem == 4: P[i, j] = np.random.randint(xl, xu)
    
        # update objective values of population    
        O = []
        for i in range(ps):
            if problem == 1: O.append(DCMaTS(P[i], m, n, pins))
            if problem == 2: O.append(GMOTS(P[i], m, n, pins))
            if problem == 3: O.append(MCut(A, P[i], m, n))
            if problem == 4: O.append(MCov(P[i], values, setlist))
        O = np.array(O)
    
        # update current best solution set
        cbest_idx, cbest_soi = Greedy(O, k, m, scalar)
        cbest = P[cbest_idx]
        cbest_o = O[cbest_idx]     
        
        # update global best solution set
        if cbest_soi < gbest_soi:
            gbest = np.copy(cbest)
            gbest_o = np.copy(cbest_o)
            gbest_soi = np.copy(cbest_soi)
             
        print("Iteration:", g, "------>", "SOI:", gbest_soi)
    
    soi_l = SOI_L(gbest_o, m)
    soi_t = SOI_T(gbest_o, m)
    runtime = time() - runtime    
    print("GEDA: SOI_L is %.2e, SOI_T is %.2e, Runtime is %.2e" % (soi_l, soi_t, runtime))





