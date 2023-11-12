# OptionPricing

<p align="center">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
</p>
 📍 Overview

The project was developed as part of the university course Computational Finance. It enables the fair pricing of financial derivatives, precisely options. 

---

 ⚙️ Features

| Feature                | Description                                                                                                                                                    |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **⚙️ Option types**    | Classical vanilla options and barrier options. |
| **📖 Exercise typ**   | European style and american style.                                       |
| **🔗 Model type**    | Black Scholes model and Heston model.                                                            |
| **🧩 Solving method**      | Analytical solution, Monte-Carlo, PDE-method, Laplace-transformation                     |


---

## 🚀 Getting Started


###  Import module
0. Import the OptionPricing module:
```sh
from calculations import BlackScholesMarket, HestonModel
```

1. Create an instance:
```sh
# default parameters
t = 0
S0 = 110
S_t = S0
r = 0.05
sigma = 0.3
T = 1
K = 100
M = 1000
sigma_tilde = 0.5
lambda_parameter = 2.5
m = 500
kappa = 0.5
gamma0 = np.square(0.2)
option_pricing_BS = BlackScholesMarket(t, S0, S_t, r, sigma, T, K, M)

```

2. Calculate fair price of a european call / put option:
```sh
V_0_call = option_pricing_BS.BlackScholes_EuCall() # call option
V_0_put = option_pricing_BS.BlackScholes_EuPut() # put option
```
3. Calculate the fair price of an american option:
```sh
V_0_call = option_pricing_BS.Am_Option_BS_LS("Call") # call option
V_0_put = option_pricing_BS.Am_Option_BS_LS("Put") # put option
```
---
## 🗺 Overview
Analytical solution: In some cases there exists are closed-form solution e.g. in the Black-Scholes model for vanilla options or even barrier options.



---

## 📄 License

This project is licensed under the `ℹ️  GNU General Public License v2.0` License. See the [LICENSE](https://github.com/timkib/OptionPricing/blob/main/LICENSE) file for additional info.

---
