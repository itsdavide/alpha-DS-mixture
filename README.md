# alpha-DS-mixture
Calibration procedure for the alpha-DS-mixture model

Author: Davide Petturiti

Affiliation: Department of Economics, University of Perugia, Italy

E-mail: davide.petturiti@unipg.it

# Reference paper
Calibration procedure for the alpha-DS-mixture model presented in the paper:
 
Davide Petturiti, Barbara Vantaggi.
No-arbitrage pricing with α-DS mixtures in a market with bid-ask spreads.
Proceedings of the Thirteenth International Symposium on Imprecise Probability: Theories and Applications, PMLR 215:401-411, 2023.
Download link: https://proceedings.mlr.press/v215/petturiti23a.html

# Bonmin solver
The Bonmin solver is required: it should be downloaded from https://www.coin-or.org/Bonmin/ and positioned in the folder ./solvers

# Libraries
Execution requires Python 3.10 and the following libraries:
* numpy 1.24.3
* pandas 2.0.0
* Pyomo 6.6.1
* more-itertools 8.12.0
* matplotlib 3.7.1

# Datasets
Folder ./datasets contains three datasets in csv format for the META stock, downloaded from Yahoo finance!:
* META_calls_2023_02_24.csv contains the bid-ask prices of call options on META at the date 2023-01-23 for the maturity 2023-02-24
* META_puts_2023_02_24.csv contains the bid-ask prices of put options on META at the date 2023-01-23 for the maturity 2023-02-24
* META_Stock_1y_2023_01_23.csv contains the META stock price time series from 2022-01-24 to 2023-01-23

# Script alpha_DS_calibration.py
The script alpha_DS_calibration.py contains the function optimal_m(n, alpha, R, file_stock, file_calls, file_puts).
Such function returns the optimal Mobius inverse minimizing the squared error between no-arbitrage alpha-DS-mixture prices and market alpha-mixture prices of puts and calls.

INPUT:
* n: number of future values for the stock random variable
* alpha: parameter of the alpha-mixtures
* R: risk-free return over the period
* file_stock: file name of the stock prices dataset
* file_calls: file name of the call option prices dataset
* file_puts: file name of the put option prices dataset

OUTPUT:
* opt_E: optimal squared error
* opt_m: optimal Mobius inverse
* d_pw: dictionary for the conversion index-subset

# Script META_alpha_0_7.py
The script META_alpha_0_7.py executes a calibration taking alpha=0.7

# Script META_alpha_tuning.py
The script META_alpha_tuning.py executes a tuning of alpha and stores the graph of the normalized squared error in the folder ./images

