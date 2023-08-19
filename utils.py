import numpy as np

def characteristic_function_BS(u, S0, r, sigma, T):
    """Returns the characteristic function in the Black-Scholes model."""
    p1 = 1j * u * (np.log(S0) + r*T)
    p2 = (1j + u**2) * sigma**2/2 *T
    return np.exp(p1 - p2)

def characteristic_function_Heston(u, S0, r, sigma_tilde, lambda_parameter, kappa, gamma0, T):
    """Returns the characteristic function in the Heston model."""
    d = np.sqrt(lambda_parameter**2 + sigma_tilde**2 * (1j * u + u**2))
    v1 = np.cosh(d * T / 2) + lambda_parameter * np.sinh(d * T / 2) / d
    v2 = (1j * u + u**2) * np.sinh(d * T / 2) / d
    v3 = np.exp(lambda_parameter * T / 2)
    v4 = np.exp(1j * u * (np.log(S0) + r * T))
    return v4 * np.power(v3 / v1, 2 * kappa / sigma_tilde**2) * np.exp(gamma0 * v2 / v1)

def f_tilde(K, z):
    """Laplace transformed payout function."""
    return K**(1 - z) / (z*(z-1))
