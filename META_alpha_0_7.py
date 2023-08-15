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

from alpha_DS_calibration import optimal_m

# Consider a 5-nomial model
n = 5 
print('Number of future values:')
print('n = ', n, '\n')

# Fix alpha
alpha = 0.7

# Risk-free interest rate per annum
tau = 32/365
r = 0.0469
R = (1 + r)**tau

(opt_E, opt_m, d_pw) = optimal_m(n, alpha, R, 'META_Stock_1y_2023_01_23.csv', 'META_calls_2023_02_24.csv', 'META_puts_2023_02_24.csv')

print('REMARK: For convenience, states of the world are indexed starting from 0, so Omega = {0, ..., n-1}\n')
    
print('Minimum squared error value = ', opt_E, '\n')

print('Optimal Mobius inverse:')
for i in range(len(d_pw)):
    print('m(', set(d_pw[i]), ') = ', round(opt_m[i], 4))