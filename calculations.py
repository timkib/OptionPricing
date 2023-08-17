### methods to calculate option price
import numpy as np
from scipy.stats import norm
from numba import jit


class OptionPricing:
    def __init__(self, t, S0, S_t, r, sigma, T, K, M):
        self.t = t
        self.S0 = S0
        self.S_t = S_t
        self.r = r
        self.sigma = sigma
        self.T = T
        self.K = K
        self.M = M
        self.H = None # Barrier
        self.d_1 = None
        self.d_2 = None
        self.d_calculator()

    # Black Scholes eu call formula
    def d_calculator(self):
        """Calculates d_1 and d_2 which is required for the Black-Scholes formula."""
        self.d_1 = (np.log(self.S_t / self.K) + (self.r + np.square(self.sigma) / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t))
        self.d_2 = self.d_1 - self.sigma * np.sqrt(self.T - self.t)

    def BlackScholes_EuCall(self):
        return self.S_t * norm.cdf(self.d_1) - self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(self.d_2)
    
    # Black Scholes eu put formula
    def BlackScholes_EuPut(self):
        return self.K * np.exp(-self.r * (self.T - self.t)) * norm.cdf(-self.d_2) -  self.S_t * norm.cdf(-self.d_1)
    
    def BlackScholes_Eu_Barrier_Call(self):
        v1 = norm.cdf((np.log(self.H**2 / self.K * self.S_t) + (self.r+self.sigma**2 / 2) * (self.T - self.t)) / (self.sigma * np.sqrt(self.T - self.t)))
        v2 = (self.H / self.S_t)**(1+2*self.r/self.sigma**2) * v1
        v3 = np.exp(-self.r * (self.T - self.t)) * self.K * self.d_1
        v4 = (self.H / self.S_t)**(2*self.r/self.sigma**2 - 1) * v1
        return self.S_t * self.d_1 - v2 - v3 - v4
        
    # Monte Carlo
    # def f(self, simulated_price, K):
        # return np.maximum(simulated_price - K, 0)

    # @jit(nopython=True, parallel=True)    
    def Eu_Option_BS_MC (self, excercise):
        simulation_array = np.zeros(shape=(self.M, ))
        for i in range(self.M):
            standard_normal_rv = np.random.randn()
            simulated_price = self.S0 * np.exp((self.r - np.square(self.sigma) / 2) * self.T + self.sigma * np.sqrt(self.T) * standard_normal_rv)
            simulated_option_price = np.maximum(simulated_price - self.K, 0) if excercise == "Call" else np.maximum(self.K - simulated_price, 0)
            simulation_array[i] = simulated_option_price
        simulation_array = simulation_array * np.exp(- self.r * self.T)
        simulated_mean_price = np.mean(simulation_array)
        # simulated_std = np.std(simulation_array, ddof=1)
        
        # Calculate 95% confidence interval
        # confidence_interval = [simulated_mean_price - 1.96 * simulated_std / np.sqrt(M), simulated_mean_price + 1.96 * simulated_std / np.sqrt(M)]
        return simulated_mean_price #, confidence_interval
    
    # @jit(nopython=True, parallel=True)
    def Eu_Option_BS_MC_Barrier(self, exercise_type, N=5_000):
        simulation_array = np.zeros(shape=(self.M, ))
        delta_t = self.T / N
        for i in range(self.M):
            simulated_path = np.zeros(shape=(N, ))
            simulated_path[0] = self.S0
            for j in range(1, N, 1):
                standard_normal_rv = np.random.randn()
                simulated_gbm = simulated_path[j - 1] * np.exp((self.r - np.square(self.sigma) / 2) * delta_t + self.sigma * standard_normal_rv * np.sqrt(delta_t))
                simulated_path[j] = simulated_gbm
                # check barrier:
                if simulated_gbm <= self.H:
                    simulation_array[i] = 0
                    break
                else:
                    simulation_array[i] = simulated_gbm
            # calculate payout
            simulation_array[i] = np.maximum(simulation_array[i] - self.K, 0) if exercise_type == "Call" else np.maximum(self.K - simulation_array[i], 0)
            
        simulation_array = simulation_array * np.exp(- self.r * self.T)
        simulated_mean_price = np.mean(simulation_array)

        return simulated_mean_price
    
    # PDE Methods
    def Eu_Option_BS_PDE(self, boundary_conditon = True):
        """PDE-method, currently with the implicit scheme.
        Returns the stock price matrix and the options prices at t=0."""
        # define parameters
        a = -0.7
        b = 0.4
        m = 500
        v_max = 2000
        q = 2*self.r / np.square(self.sigma)
        delta_x = (b-a)/m
        delta_t = np.square(self.sigma) * self.T / (2 * v_max)
        lambda_ = delta_t / np.square(delta_x)

        price_matrix = np.zeros(shape=(m + 1,m + 1))
        final_stock_prices = np.zeros(shape=(m + 1))
        w = np.zeros(shape=(m + 1))
        # Add stock prices for the first row:
        for i in range(m + 1):
            # price_matrix[m - i - 1, 0] = K * np.exp(a + i * delta_x)
            final_stock_prices[i] =  b - i * delta_x
            
        # calculate payout for the first row:
        for i in range(m + 1):
            x_i = final_stock_prices[i]
            wi = np.maximum(np.exp(x_i*0.5*(q+1)) - np.exp(x_i * 0.5 * (q-1)), 0)
            price_matrix[i, 0] = wi
            w[i] = wi
            
        A_impl = np.zeros(shape=(m + 1,m + 1))
        A_impl[0, 0] = 1 + 2*lambda_
        A_impl[0, 1] = - lambda_
        A_impl[-1, -2] = - lambda_
        A_impl[-1, -1] = 1 + 2*lambda_

        for i in range(m-1):
            A_impl[i + 1, i] = - lambda_
            A_impl[i + 1, i + 1] = 1 + 2 * lambda_
            A_impl[i + 1, i + 2] = - lambda_
        
        price_matrix[:, 0] = w
        A_impl_inv = np.linalg.inv(A_impl)
        # get the option prices
        for i in range(1, m + 1):
            t = i * delta_t
            x = final_stock_prices[0]
            price_matrix[:, i] = np.linalg.matrix_power(A_impl_inv, i) @ w
            # with boundary conditions
            if boundary_conditon == True: 
                price_matrix[0, i] = np.exp(0.5 * (q+1)*x + 0.25 * (q+1)**2 * t) - np.exp(0.5*(q-1)*x + 0.25*(q-1)**2 * t)
                price_matrix[-1, i] = 0
        
        # transform back
        S_matrix = np.zeros(shape=(m + 1))
        v_0 = np.zeros(shape=(m + 1))

        for i in range(m + 1):
            x_i = final_stock_prices[i]
            S_matrix[i] = self.K * np.exp(x_i)
            v_0[i] = self.K * price_matrix[i, -1] * np.exp(-x_i/2*(q-1)-0.5*self.sigma**2*self.T*(q+0.25*(q-1)**2))
        return S_matrix, v_0 
    
