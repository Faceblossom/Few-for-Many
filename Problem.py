import numpy as np




def DCMaTS(x, m, n, pins):
    f = np.zeros(m)
    for i in range(m):
        if pins == 1:
            f[i] = - x[i] + (np.sum(x) - x[i]) / (n - 1)
        elif pins == 2:
            f[i] = - x[i]**2 + (np.sum(x**2) - x[i]**2) / (n - 1)
        elif pins == 3:
            f[i] = - x[i]**0.5 + (np.sum(x**0.5) - x[i]**0.5) / (n - 1)
        elif pins == 4: 
            f[i] = - np.sin(0.5*np.pi*x[i]) + (np.sum(np.sin(0.5*np.pi*x)) - np.sin(0.5*np.pi*x[i])) / (n - 1)
    return f




def GMOTS(x, m, n, pins):
    g = 0
    for i in range(m, n):
        g += (1 + (x[i] - 0.5)**2 - np.cos(20*np.pi*(x[i] - 0.5)))
    g = g*100
    f = np.zeros(m)
    for i in range(m):
        u = x[i]
        v = (np.sum(x[:m]) - x[i]) / (m - 1)
        if pins == 1: f[i] = 0.5 * (u - v + 1) * (1 + g)
        if pins == 2: f[i] = (0.5 * (u - v + 1))**0.5 * (1 + g)
        if pins == 3: f[i] = (0.5 * (u - v + 1))**2 * (1 + g)
        if pins == 4: f[i] = (2**(0.5*(u - v + 1)) - 1) * (1 + g)
        if pins == 5: f[i] = np.sin(0.25 * np.pi * (u - v + 1)) * (1 + g)
        if pins == 6: f[i] = 2 / np.pi * np.arcsin(0.5 * (u - v + 1)) * (1 + g)
    return f




def MCut_Init(m):
    
    A0 = np.loadtxt('Data-AF-A.txt')
    n = np.size(A0, 0)
    
    np.random.seed(100)
    
    A = np.zeros([n, n, m])
    
    for i in range(n):
        for j in range(i + 1, n):
            if A0[i, j] == 1:
                A[i, j] = A[j, i] = np.random.random([m])
    
    np.random.seed(None)

    return A, n


def MCut(A, x, m, n):
    
    mcut_value = np.zeros(m)

    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                mcut_value += A[i][j]
                
    return -mcut_value




def MCov_Init(m):

    A = np.loadtxt('Data-NS-A.txt')
    n = np.size(A, 0)
    
    np.random.seed(100)
    
    value = np.random.random([n, m])
    
    np.random.seed(None)
    
    setlist = []
    for i in range(n):
        seti = [i]
        for j in range(n):
            if A[i, j] == 1: seti.append(j)
        setlist.append(seti)
    
    return value, setlist, n


def MCov(x, values, setlist):
    
    x_union = set()
    for i in x:
        buf = set(setlist[i])
        x_union = x_union.union(buf)
    mcov_value = np.sum(values[list(x_union)], 0)
                
    return -mcov_value



