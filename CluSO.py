import numpy as np
from time import *
from sklearn.cluster import KMeans


from Problem import DCMaTS
from Problem import GMOTS


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
    
    problem = 2 # 1: DC-MaTS; 2: GMOTS
    scalar = 1 # scalarization method 1: linear; 2: Tchebycheff 

    pins = 1 # problem instance
    k = 5 # number of solutions
    m = 50 # number of objectives
    n = 75 # variable dimension
    xl = 0 # lower bound of variable
    xu = 1 # upper bound of variable
    ps = 20*m # population size
    ic = 20*m # iteration count 
    alpha, beta = 1, 1 #parameters
      
    runtime = time()
    
    # initialize position (population) and velocity
    P = np.random.uniform(xl, xu, (ps, n))
    V = np.zeros([ps, n])
    
    # objective values of population
    O = []
    for i in range(ps):
        if problem == 1: O.append(DCMaTS(P[i], m, n, pins))
        if problem == 2: O.append(GMOTS(P[i], m, n, pins))
    O = np.array(O)
    
    # label of objectives
    km = KMeans(n_clusters = k, random_state = 1000)
    km.fit(O.T)
    L = km.labels_
    
    # clusters of objectives
    C = []
    for i in range(k):
        C.append(list(np.where(L == i)[0]))        
    
    # objective cluster values of population 
    O_cluster = np.zeros([ps, k])
    for i in range(ps):
        for j in range(k):
            O_cluster[i][j] = sum(O[i][C[j]])
    
    # initialize current best solution set
    cbest_idx, cbest, cbest_o, cbest_soi = [], [], [], 0
    for j in range(k):
        Ocj_min = np.min(O_cluster[:, j])
        idx = np.where(O_cluster[:, j] == Ocj_min)[0]
        if np.size(idx) > 1: idx = np.random.choice(idx)
        else: idx = idx[0]
        cbest_idx.append(idx)
        cbest.append(P[idx])
        cbest_o.append(O[idx])
        cbest_soi += Ocj_min / m
    cbest = np.array(cbest)
    cbest_o = np.array(cbest_o)
    
    # initialize global best solution set
    gbest = np.copy(cbest)
    gbest_o = np.copy(cbest_o)
    gbest_soi = np.copy(cbest_soi)

    for g in range(ic):
        
        for i in range(ps):
            if i in cbest_idx: continue          
            Oci_min = np.min(O_cluster[i])
            idx = np.where(O_cluster[i] == Oci_min)[0] # index of the most suitable solution in cbest/gbest for P[i] 
            if np.size(idx) > 1: idx = np.random.choice(idx)
            else: idx = idx[0]
            R = np.random.uniform(xl, xu, (3, n))
            V[i] = R[0] * V[i] + alpha * R[1] * (cbest[idx] - P[i]) + beta * R[2] * (gbest[idx] - P[i])
            P[i] = P[i] + V[i]
            for j in range(n):
                if P[i, j] < xl or P[i, j] > xu: P[i, j] = np.random.uniform(xl, xu) # repair

        # update objective values of population    
        O = []
        for i in range(ps):
            if problem == 1: O.append(DCMaTS(P[i], m, n, pins))
            if problem == 2: O.append(GMOTS(P[i], m, n, pins))
        O = np.array(O)
        
        # update objective cluster values of population
        O_cluster = np.zeros([ps, k])
        for i in range(ps):
            for j in range(k):
                O_cluster[i][j] = sum(O[i][C[j]])
                
        # update current best solution set
        cbest_idx, cbest, cbest_o, cbest_soi = [], [], [], 0
        for j in range(k):
            Ocj_min = np.min(O_cluster[:, j])
            idx = np.where(O_cluster[:, j] == Ocj_min)[0]
            if np.size(idx) > 1: idx = np.random.choice(idx)
            else: idx = idx[0]
            cbest_idx.append(idx)
            cbest.append(P[idx])
            cbest_o.append(O[idx])
            cbest_soi += Ocj_min / m
        cbest = np.array(cbest)
        cbest_o = np.array(cbest_o)
        
        # update global best solution set
        if cbest_soi < gbest_soi:
            gbest = np.copy(cbest)
            gbest_o = np.copy(cbest_o)
            gbest_soi = np.copy(cbest_soi)
        
        # update label of objectives
        L = []
        for j in range(m):
            min_value = np.min(gbest_o[:, j])
            idx = np.where(gbest_o[:, j] == min_value)[0]
            if np.size(idx) > 1: L.append(np.random.choice(idx))
            else: L.append(idx[0])
        L = np.array(L)  
        
        # update clusters of objectives
        C = []
        for i in range(k):
            C.append(list(np.where(L == i)[0]))          
             
        print("Iteration:", g, "------>", "SOI:", gbest_soi)
    
    soi_l = SOI_L(gbest_o, m)
    soi_t = SOI_T(gbest_o, m)
    runtime = time() - runtime    
    print("CluSO: SOI_L is %.2e, SOI_T is %.2e, Runtime is %.2e" % (soi_l, soi_t, runtime))





