import numpy as np

def sim_gbm_paths(S0, r, sigma, T, N, m):
    """Simulates N different geometric brownian motion pathes with m equidistant pathes."""
    sim_pathes = np.zeros(shape=(N, m))
    sim_pathes[:, 0] = S0
    dt = T / m
    for i in range(N):
        brownian_motion = np.random.normal(loc=0.0, scale=1.0, size=m-1) 
        sim_pathes[i, 1:] = S0 * np.cumproduct(np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * brownian_motion))
    
    return sim_pathes