import numpy as np

def characteristic_function_BS(u, S0, r, sigma, T):
    """Returns the characteristic function in the Black-Scholes model."""
    p1 = 1j * u * (np.log(S0) + r*T)
    p2 = (1j + u**2) * sigma**2/2 *T
    return np.exp(p1 - p2)

def f_tilde(K, z):
    """Laplace transformed payout function."""
    # check condition:
    return K**(1 - z) / (z*(z-1))
