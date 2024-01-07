# Long Short Trading Strategy on US ETFs QQQ & IWM

## Background (Some qualitative reasons)
Near the end of 2023, after the FED raised their expected rate cuts in the coming years, mid/small caps outperformed the big cap companies as many believe lower interest rates tend to benefit small companies more. In 23Q4, $IWM outperformed $QQQ. It then made me wonder if a long short portfolio works well in the recent years relative to only $QQQ or $IWM, and if this long-short portfolio outperforms the individual ETF.

## Data Sampling
Traning and test (validation) set from [yfinance](https://pypi.org/project/yfinance/), a commonly-used Python library (2022-23 hourly data throughout)

_Only the hourly data within 730 days can be downloaded from `yfinance` at no cost and I fetched the data at 2023 year end. More granular historical hourly data is a paid resource elsewhere._

## Parameters (# of epochs, hidden layers, neurons, the learning rate of Adam optimizers)
Brute force training the given data (2/3 as training and the rest as test), pick the set of parameters that maximizes the Sharpe ratio of the portfolio performance in the test set period after excluding outliers.
> Result: QQQ_weight = -0.3, IWM_weight = 1.3, hidden_layers = 4, neurons = 60, Adam_learning_rate = 0.3

## DNN model: More details
- Dropout rate 0.3 in between each layer
- L1 regularization (λ = 0.0005)
- Loss function: Binary Cross Entropy
- Optimizer: Adam (Adaptive Moment Estimation)
- Class weight adjustment: tell the model to pay more attention (heavier weight) on the relativity few cases that are available. Usually this technique is used on imbalanced data, as suggested in the [official Tensorflow tutorials](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#class_weights).

## Indicators as features 
SMA(50) - SMA(150), EMA, Boillinger bands, RSI of 5/10 days, MACD, KD lines, stochastic oscillator, parabolic SAR, Aroon Indicator.

_I tried to include 30+ features and fit a model on them. Now I am still learning how to improve the models by adjusting some params or removing some features._

## Portfolio
Base portfolio: 130% of IWM & -30% of QQQ (opposite position of IWM)

Long or short position is based on the base portfolio. E.g. if the decision is to go long position in the base portfolio, we take 130% long position in IWM and short 30% of QQQ in terms of the net worth.

## Model prediction output & loss function
The model predicts a binary outcome: if the portfolio will rise in the next hour or not. The output is therefore a probablity of portfolio appreciation in the next hour.

## Decision making on LONG or SHORT
Based on the prediction on test set, select 20<sup>th</sup> and 80<sup>th</sup> percentile of the prediciton results on the test set as the lower and upper decision boundaries. Usually, and as I expect, the prediction probablity distribution will center at around 0.5.
1. If the prediction on the latest tick data is in between, maintain the position. **(No new market order placed)**
2. Else if the prediction is below the lower bound, short the base portfolio (by setting the desired position to -1)
3. Else, long the base portfolio.

## Trading API
[IBKR Trader workstation (TWS)](https://www.interactivebrokers.com/en/trading/tws.php) API with [ib_insync](https://pypi.org/project/ib-insync/) API wrapper.

The connection is established via active TWS login session hence no explict token is needed.

## Streamed Market Data
#### US Equities and ETFs
Subscription required and only available in live trading accounts (non-paper). Paper accounts can only request 15m-delayed data and is not streamed (therefore not automated unless we fetch the live data somewhere else, e.g. `yfinance`).
#### Forex
Complete live and streaming enabled without any subscription, even in paper account.

## Implementation & Execution
Run the `implementation_tool.py` as a script. (Make sure the TWS is logged in.)

All orders placed are market orders.

## Trade reports (Some SWE stuff)
[Broadcast](https://developers.line.biz/en/reference/messaging-api/#send-broadcast-message) a simple summary of the trades to an **LINE channel (official account)** after each trading day.

Broadcast info: TBD

Making Line Messaging API call: Called at the end of trading sessions, abstraction done.

## Existing problems
1. The timestamps of hourly data differ between `yfinance` and `ib_insync`. From `yfinance`, the quotes are snapshot at 9:30, 10:30, ..., 15:30 EST, whereas the data from IB API is at `k` o'clock sharp. Any underlying distribution in the time series should be the same despite this fact.
2. Non-live streaming data in paper account.
3. Fixed weights on QQQ/IWM so the strategy is kinda rigid.
4. The market orders in paper trading usually take up to a few minutes to be filled even for QQQ such a highly liquid security (don't know why). Making tracking the portfolio net worth harder. 

## To do list
- Broadcasting summary formatting `pandas.DataFrame.to_json(orient = "index")` (Details TBC)
- Line API connection and callback code (a separate file that wouldn't be pushed to github)
- Some error handling (suggested)
- Line messages formatting. See [Flex message](https://developers.line.biz/en/docs/messaging-api/flex-message-elements/) for more. (optional)
- Train a new model on "USDJPY" & "EURUSD" (replicate what's happening in QQQ/IWM context but there is no background reason.) (optional)
- Helper functions generalized to FOREX cfd. _(For forex cfd trades, need to create two separate contracts: cash trading pairs for data, cfd for orders)_ (optional)
- Making the code OO to handle global variables (by making them instance variables, optional)

### Credits
Partial codes are modified from the resources of a Udemy course.

https://www.udemy.com/course/algorithmic-trading-with-python-and-machine-learning/
