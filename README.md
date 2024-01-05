# Long Short Trading Strategy on US ETFs QQQ & IWM

## Data Sampling
Traning and test (validation) set from [yfinance](https://pypi.org/project/yfinance/), a commonly-use Python library (2022-23 hourly data throughout)

_Only the hourly within 730 days can be downloaded at no cost and I fetched the data at 2023 year end. More granular historical hourly data is a paid resource._

## Parameters (# of epochs, hidden layers, neurons, and the learning rate of Adam optimizers)
Brute force training the given data (2/3 as training and the rest as test), pick the set of parameters that maximizes the Sharpe ratio of the portfolio performance in the test set period after excluding outliers.
> Result: QQQ_weight = -0.3, IWM_weight = 1.3, hidden_layers = 4, neurons = 60, Adam_learning_rate = 0.3

## DNN model: More details
- Dropout rate 0.3 in between each layer
- L1 regularization (λ = 0.0005)
- Loss function: Binary Cross Entropy
- Optimizer: Adam (Adaptive Moment Estimation)

## Portfolio
Base portfolio: 130% of IWM & -30% of QQQ (opposite position of IWM)
Long or short position is based on the base portfolio. E.g. if the decision is to go long position in the base portfolio, we take 130% long position in IWM and short 30% of QQQ in terms of the net worth.

## Model prediction output & loss function
The model predicts a binary outcome: if the portfolio will rise in the next hour or not. The output is therefore a probablity 

## Decision making on LONG or SHORT
Based on the prediction on test set, select 20<sup>th</sup> and 80<sup>th</sup> percentile of the prediciton results on the test set as the lower and upper decision boundaries. Usually, and as I expect, the prediction probablity distribution will center at around 0.5.
1. If the prediction on the latest tick data is in between, maintain the position. **(No new market order placed)**
2. Else if the prediction is below the lower bound, short the base portfolio (by setting the desired position to -1)
3. Else, long the base portfolio.

## Trading API
[IBKR Trader workstation (TWS)](https://www.interactivebrokers.com/en/trading/tws.php) API with [ib_insync](https://pypi.org/project/ib-insync/) API wrapper. The connection is established via active TWS login session hence no explict token is needed.

## Streamed Market Data
#### US Equities and ETFs
