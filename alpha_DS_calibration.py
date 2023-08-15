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

import pandas as pd
import numpy as np
import pyomo.environ as pyo
from more_itertools import powerset

##############################################################################
##############################################################################

def call_payoff(i, K, S1):
    return max(S1[i] - K, 0)

def min_call(S, K, S1):
    min_val = max(S1)
    for i in S:
        if call_payoff(i, K, S1) < min_val:
            min_val = call_payoff(i, K, S1)
    return min_val

def max_call(S, K, S1):
    max_val = 0
    for i in S:
        if call_payoff(i, K, S1) > max_val:
            max_val = call_payoff(i, K, S1)
    return max_val

def put_payoff(i, K, S1):
    return max(K - S1[i], 0)

def min_put(S, K, S1):
    min_val = max(S1)
    for i in S:
        if put_payoff(i, K, S1) < min_val:
            min_val = put_payoff(i, K, S1)
    return min_val

def max_put(S, K, S1):
    max_val = 0
    for i in S:
        if put_payoff(i, K, S1) > max_val:
            max_val = put_payoff(i, K, S1)
    return max_val

##############################################################################
##############################################################################

# Function that returns the optimal Mobius inverse minimizing the squared error
# of no-arbitrage alpha-DS-mixture prices and market alpha-mixture prices between
# puts and calls
# 
# INPUT
# n: number of future values for the stock random variable
# alpha: parameter of the alpha-mixtures
# R: risk-free return over the period
# file_stock: file name of the stock prices dataset
# file_calls: file name of the call option prices dataset
# file_putss: file name of the put option prices dataset
#
# OUTPUT
# opt_E: optimal squared error
# opt_m: optimal Mobius inverse
# d_pw: dictionary for the conversion index-subset

def optimal_m(n, alpha, R, file_stock, file_calls, file_puts):
    epsilon = 0.0001
    
    # Payoff
    STOCK_stock = pd.read_csv('./datasets/' + file_stock)['Close']
    print('Creating the stock future range:')
    S1_min = round(min(STOCK_stock), 0)
    S1_max = round(max(STOCK_stock), 0)
    step = (S1_max - S1_min) / n
    S1_range = np.arange(S1_min, S1_max+1, step)
    S1 = [round((S1_range[i+1] + S1_range[i]) / 2, 1) for i in range(n)]
    print('S1 =', S1, '\n')
    
    # Generate the power set
    I_singletons = list(range(n))
    pw = [list(x) for x in powerset(I_singletons)]
    pw.pop(0)
    
    # Index the sets in the power set and create a dictionary
    I_vars = list(range(len(pw)))
    d_pw = dict(zip(I_vars, pw))
    
    
    # Load STOCK calls
    STOCK_calls = pd.read_csv('./datasets/' + file_calls)[['strike','bid','ask']]
    STOCK_calls['alpha_p'] = alpha * STOCK_calls['bid'] + (1 - alpha) * STOCK_calls['ask']
    
    # Load STOCK puts
    STOCK_puts = pd.read_csv('./datasets/' + file_puts)[['strike','bid','ask']]
    STOCK_puts['alpha_p'] = alpha * STOCK_puts['bid'] + (1 - alpha) * STOCK_puts['ask']
    
    I_calls = list(range(len(STOCK_calls)))
    I_puts = list(range(len(STOCK_puts)))
    
    
    # Generate the alpha-gambles for the calls
    alpha_gamble_calls = {}
    for i in I_calls:
        K = STOCK_calls.iloc[i]['strike']
        for j in I_vars:
            alpha_gamble_calls[i, j] = round((alpha * min_call(d_pw[j], K, S1) + (1 - alpha) * max_call(d_pw[j], K, S1)), 6)
    
    # Generate the alpha-gambles for the puts
    alpha_gamble_puts = {}
    for i in I_puts:
        K = STOCK_puts.iloc[i]['strike']
        for j in I_vars:
            alpha_gamble_puts[i, j] = round((alpha * min_put(d_pw[j], K, S1) + (1 - alpha) * max_put(d_pw[j], K, S1)), 6)
    
    # Create the optimization model
    model = pyo.ConcreteModel()
    model.I_vars = pyo.Set(initialize=I_vars)
    model.I_calls = pyo.Set(initialize=I_calls)
    model.I_puts = pyo.Set(initialize=I_puts)
    
    # Create parameters in the model
    model.alpha_price_calls = pyo.Param(model.I_calls, initialize=round(STOCK_calls['alpha_p'],6))
    model.alpha_price_puts = pyo.Param(model.I_puts, initialize=round(STOCK_puts['alpha_p'],6))
    model.alpha_gamble_calls = pyo.Param(model.I_calls, model.I_vars, initialize=alpha_gamble_calls)
    model.alpha_gamble_puts = pyo.Param(model.I_puts, model.I_vars, initialize=alpha_gamble_puts)
    
    # Create variables for the optimization task
    def BoundsIntializer(model, i):
        return (0 if I_singletons.count(i) == 0 else epsilon, None)
    
    model.m = pyo.Var(model.I_vars, within=pyo.NonNegativeReals, bounds=BoundsIntializer)
    model.ec = pyo.Var(model.I_calls, within=pyo.NonNegativeReals)
    model.ep = pyo.Var(model.I_puts, within=pyo.NonNegativeReals)
    model.err = pyo.Var(within=pyo.NonNegativeReals)
    
    # Set the objective
    model.o = pyo.Objective(expr = model.err, sense=pyo.minimize)
    
    
    ##############################################################################
    ##############################################################################
    # Add constraints
    
    model.c_norm = pyo.Constraint(expr=sum(model.m[i] for i in model.I_vars) == 1)
    
    ##############################################################################
    
    def ConstrRuleCall(model, i):
        return model.ec[i] == (sum(model.alpha_gamble_calls[i, j] * model.m[j] for j in model.I_vars) / R - model.alpha_price_calls[i])**2
    
    model.c_calls = pyo.Constraint(model.I_calls, rule=ConstrRuleCall)
    
    ##############################################################################
    
    def ConstrRulePut(model, i):
        return model.ep[i] == (sum(model.alpha_gamble_puts[i, j] * model.m[j] for j in model.I_vars) / R - model.alpha_price_puts[i])**2
    
    model.c_puts = pyo.Constraint(model.I_puts, rule=ConstrRulePut)
    
    ##############################################################################
    
    model.c_err = pyo.Constraint(expr=model.err == sum(model.ec[i] for i in model.I_calls) + 
                                sum(model.ep[i] for i in model.I_puts))
    
    ##############################################################################
    ##############################################################################
    
    # Solve the problem
    status = pyo.SolverFactory('./solvers/bonmin').solve(model)
    pyo.assert_optimal_termination(status)
    
    # Get the optimal squared error
    opt_E = pyo.value(model.o)
    
    # Get the optimal Mobius inverse
    opt_m = {i : pyo.value(model.m[i]) for i in model.I_vars}
    
    # Return the optimal squared error, the optimal Mobius inverse and the dictionary
    # for the conversione index-subset
    return (opt_E, opt_m, d_pw)