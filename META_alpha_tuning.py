#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Davide Petturiti 
Affiliation: Department of Economics, University of Perugia, Italy
E-mail: davide.petturiti@unipg.it


Calibration procedure for the alpha-DS-mixture model presented in the paper:
 
Davide Petturiti, Barbara Vantaggi.
No-arbitrage pricing with α-DS mixtures in a market with bid-ask spreads.
Proceedings of the Thirteenth International Symposium on Imprecise Probability:
Theories and Applications, PMLR 215:401-411, 2023.
Download link: https://proceedings.mlr.press/v215/petturiti23a.html

"""

import numpy as np
from alpha_DS_calibration import optimal_m
import matplotlib.pyplot as plt

# Consider a 5-nomial model
n = 5 
print('Number of future values:')
print('n = ', n, '\n')

# Risk-free interest rate per annum
tau = 32/365
r = 0.0469
R = (1 + r)**tau

ticker = 'META'

# Values of alpha
alpha = np.round(np.arange(0, 1.1, 0.1), 1)

# Create the vector of squared errors
errors = np.zeros(len(alpha))

print('\nTuning alpha for:', ticker, '\n')
for i in range(len(alpha)):
    print('alpha = ', alpha[i])
    (opt_E, opt_m, d_pw) = optimal_m(n, alpha[i], R, ticker + '_Stock_1y_2023_01_23.csv',
                                     ticker + '_calls_2023_02_24.csv', 
                                     ticker + '_puts_2023_02_24.csv')
    errors[i] = opt_E


# Compute the normalized errors
n_errors = (errors   - min(errors)) / (max(errors) - min(errors))

# Plot the normalized errors
plt.figure(figsize=(6, 4))
plt.title(r'Normalized optimal squared error as a function of $\alpha$')
plt.plot(alpha, n_errors, label=ticker, c='red', marker='o')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Normalized $E(\widehat{\varphi_\alpha})$')
plt.legend()
plt.savefig('./images/NormE.png', dpi=300)