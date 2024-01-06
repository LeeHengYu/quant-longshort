import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment', None)
# surpress chain assignment warning (already used loc but still not working)
# suspect false positives (according to pandas official doc)

import datetime as dt
import os

import ta
from ib_insync import *

from DNNModel import *

ib = IB()
ib.connect()

wQQQ = -0.3
wIWM = 1 - wQQQ
start_time = dt.time(14,27,0) # 3 mins before regular trading hours
end_time = dt.time(20,30,0)
cash = 10000

# qqq = Stock('QQQ', 'SMART', 'USD') # Actual implementation
# iwm = Stock('IWM', 'SMART', 'USD') # Actual implementation
qqq = Forex('USDJPY') # For testing
iwm = Forex('EURUSD') # For testing
ib.qualifyContracts(qqq, iwm)

def initialize_stream():
    global qqq_bars, iwm_bars
    qqq_bars = ib.reqHistoricalData(
        qqq,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='MIDPOINT',
        useRTH=True,
        formatDate=2,
        keepUpToDate=True)
    iwm_bars = ib.reqHistoricalData(
        iwm,
        endDateTime='',
        durationStr='1 D',
        barSizeSetting='1 min',
        whatToShow='MIDPOINT',
        useRTH=True,
        formatDate=2,
        keepUpToDate=True)
    qqq_bars.updateEvent += onQQQBarUpdate
    iwm_bars.updateEvent += onIWMBarUpdate

def start_session():
    global last_update, session_start
    last_update = dt.datetime.utcnow()
    session_start = pd.to_datetime(last_update).tz_localize("utc")

    initialize_stream()
    print("Establish Stream")
    compile_initial_data()
    print("Data complied!")
    stop_session()
    
def compile_initial_data():
    global qqq_data, iwm_data, all_data, latest_price
    qqq_data = prepare_data(qqq_bars)
    iwm_data = prepare_data(iwm_bars)
    all_data = pd.concat([qqq_data, iwm_data], axis = 1).dropna()
    latest_price = { b.contract.localSymbol.replace('.', ''): b[-1].close for b in [qqq_bars, iwm_bars] }
    os.system('clear') # 'cls' on WindowsOS
    print(all_data)
    
def stop_session(startTime_enabled = True, endTime_enabled = True):
    print("Into the while loop!")
    while True:
        if startTime_enabled and dt.datetime.utcnow().time() < start_time:
            print("Before trading hours, the session automatically stops.")
            break
        if endTime_enabled and dt.datetime.utcnow().time() >= end_time:
            try:
                report, PnL = trade_reporting()
            except:
                report = PnL = None
            if report and PnL:
                print("Data is valid for API call")
                # LINE messaging API function call (callback import from another module)
            print("End session as planned")
            break
        ib.sleep(10)
    ib.cancelHistoricalData(qqq_bars)
    ib.cancelHistoricalData(iwm_bars)
    ib.disconnect()

def prepare_data(bars): # input: BarDataList
    df = pd.DataFrame(bars).set_index('date').tz_convert('US/Eastern').iloc[:-1, 1:4]
    ticker = bars.contract.symbol
    attributes = ['High', 'Low', 'Close']
    df.columns = [ticker + '_' + att for att in attributes]
    return df

def add_features(df, ticker_list = ['QQQ', 'IWM']):
    for ticker in ticker_list:
        df[f"{ticker}_return"] = np.log(df[f"{ticker}_Close"].div(df[f"{ticker}_Close"].shift(1)))
        df.fillna(0, inplace=True)
        df[f"{ticker}_creturn"] = df[f"{ticker}_return"].cumsum().apply(np.exp)
        
    windows = [5, 10, 20, 50]
    for ticker in ticker_list:
        for window in windows:
            df[f'{ticker}_EMA_{window}'] = ta.trend.EMAIndicator(df[f'{ticker}_Close'], window = 7*window).ema_indicator()
    
    windows = [3, 5, 10]
    for ticker in ticker_list:
        for window in windows:
            bollinger_object = ta.volatility.BollingerBands(df[f'{ticker}_Close'], window = 7*window)
            df[f'{ticker}_LBol_{window}'] = bollinger_object.bollinger_lband()
            df[f'{ticker}_HBol_{window}'] = bollinger_object.bollinger_hband()
    
    for ticker in ticker_list:
        df[f'{ticker}_RSI_5'] = ta.momentum.RSIIndicator(df[f'{ticker}_Close'], window = 5*7).rsi()
        df[f'{ticker}_RSI_10'] = ta.momentum.RSIIndicator(df[f'{ticker}_Close'], window = 5*10).rsi()
        
        df[f'{ticker}_SMA_50_150'] = df[f'{ticker}_Close'].rolling(50).mean() - df[f'{ticker}_Close'].rolling(150).mean()
        
        df[f"{ticker}_MACD"] = ta.trend.MACD(df[f'{ticker}_Close'], window_fast=12*7, window_slow=24*7, window_sign=8*7).macd()
        df[f'{ticker}_MACD_hist'] = ta.trend.MACD(df[f'{ticker}_Close'], window_fast=12*7, window_slow=26*7, window_sign=9*7).macd_diff()
        
        KD_object = ta.momentum.StochasticOscillator(df[f'{ticker}_High'], df[f'{ticker}_Low'], df[f'{ticker}_Close'], window = 14*7)
        df[f'{ticker}_KD_K'] = KD_object.stoch()
        df[f'{ticker}_KD_D'] = KD_object.stoch_signal()
        
        o = ta.trend.PSARIndicator(df[f'{ticker}_High'], df[f'{ticker}_Low'], df[f'{ticker}_Close'])
        df[f'{ticker}_SAR'] = o.psar()
        
        df[f'{ticker}_AroonIndicator'] = ta.trend.AroonIndicator(df[f'{ticker}_High'], df[f'{ticker}_Low'], window = 50).aroon_indicator()
        
    return df

def fit_model(layer = 4, neurons = 60, epochs = 50, wQQQ = - 0.3):
    """
    Return the test set with the prediction.
    """
    
    model = create_model(layer, neurons, adam_LR = 0.003, input_dim = len(cols))

    add_portfolio_setting(featured, QQQ_weight = wQQQ)

    split_idx = int(len(featured) * 0.66667)

    train = featured.iloc[:split_idx, :]
    test = featured.iloc[split_idx:, :]

    mu, std = train[cols].mean(), train[cols].std()
    train_s = (train[cols] - mu) / std
    test_s = (test[cols] - mu) / std

    model.fit(x = train_s, y = train["LSportfolio_direction"], epochs=epochs, verbose=0, validation_split = 0.25, shuffle = False, class_weight = cw(featured))
    test['pred'] = model.predict(test_s)
    # Creating a new dataframe column yields false positive warning, surpressed option applied when importing pd
    low, high = np.percentile(test.pred, 20), np.percentile(test.pred, 80)
    test.loc[:,'position'] = np.where(test.pred >= high, 1, np.nan)
    test.loc[:,'position'] = np.where(test.pred <= low, -1, test.position)
    test.loc[:,'position'] = test.position.ffill().fillna(0)

    return test, low, high, model, mu, std
    # reuse some parameters after fitting the model

def predict_last(df, mu, std):
    """
    Need to run fit_model() first to store global variables
    Parse only featured columns
    """
    single_X = df.iloc[-1:, :]
    single_X = (single_X - mu) / std
    print(f'Last tick data: {single_X.index[0]}')
    return model.predict(single_X)[0][0]

def extract_features(df, t1: str, t2: str):
    # constant filtering
    features = ['_AroonIndicator', '_EMA_10', '_EMA_20', '_EMA_5', '_EMA_50', '_HBol_10', '_HBol_3', '_HBol_5', '_KD_D', '_KD_K', '_LBol_10', '_LBol_3', '_LBol_5', '_MACD', '_MACD_hist', '_RSI_10', '_RSI_5', '_SAR', '_SMA_50_150']

    cols = []
    for f in features:
        cols.append(t1 + f)
        cols.append(t2 + f)
    return df[cols]
    
def process_new_bar():
    global all_data
    # last_update = dt.datetime.utcnow()
    # print(last_update)
    if qqq_data.index[-1] == iwm_data.index[-1] and qqq_data.index[-1] > all_data.index[-1]:
        all_data = pd.concat([qqq_data, iwm_data], axis = 1)
        os.system('clear')
        print(all_data.iloc[-10:]) # print out the latest 10 ticks
        extract_features(add_features(all_data, [qqq_bars.symbol, iwm_bars.symbol]), qqq_bars.symbol, iwm_bars.symbol)
        guess_prob = predict_last(all_data, mu ,std) # model only designed for QQQ/IWM strat for now
        if low < guess_prob < high: # do nothing
            return
        elif guess_prob <= low:
            target = -1
        else:
            target = 1
        print(f'New target position = {target}')
        execute_trade(target)
             
def updateLatestPrice(bars):
    temp = bars.contract.localSymbol.replace('.', '')
    latest_price[temp] = bars[-1].close

def onQQQBarUpdate(bars, hasNewBar):
    global qqq_data
    if qqq_data.index[-1] >= bars[-1].date:
        return
    updateLatestPrice(bars)
    qqq_data = prepare_data(bars)
    print("Received QQQ data")
    process_new_bar()

def onIWMBarUpdate(bars, hasNewBar): 
    global iwm_data
    updateLatestPrice(bars)
    if iwm_data.index[-1] >= bars[-1].date:
        return
    iwm_data = prepare_data(bars)
    print("Received IWM data")
    process_new_bar()

def getPrice(ticker):
    if ticker in latest_price:
        return latest_price[ticker]
    return -1

def send_order(units, contract):
    if units > 0:
        side = 'BUY'
    elif units < 0:
        side = 'SELL'
    else: # no trades
        return
    
    order = MarketOrder(side, abs(units))
    ib.placeOrder(contract, order)

def execute_trade(target_pos: float) -> None:
    original_cash = cash
    try:
        current_IWM_pos = [pos.position for pos in ib.positions() if pos.contract.conId == iwm.conId][0]
    except:
        current_IWM_pos = 0
    try:
        current_QQQ_pos = [pos.position for pos in ib.positions() if pos.contract.conId == qqq.conId][0]
    except:
        current_QQQ_pos = 0
    
    netWorth = all_data.QQQ_Close[-1] * current_QQQ_pos + all_data.IWM_Close[-1] * current_IWM_pos + original_cash
        # calculate the latest net worth based on the latest tick price
        
    desired_QQQ_shares = netWorth * wQQQ / getPrice('QQQ') * target_pos
    desired_IWM_shares = netWorth * wIWM / getPrice('IWM') * target_pos
    reqQQQTrades, reqIWMTrades = desired_QQQ_shares - current_QQQ_pos, desired_IWM_shares - current_IWM_pos
    send_order(reqQQQTrades, qqq)
    send_order(reqIWMTrades, iwm)
    # rebalance every hour (lower volatility but more trans costs)
    # inaccurate calculation on balance as we're getting delayed prices on stocks (paper account)
    cash = netWorth - reqQQQTrades * getPrice('QQQ') - reqIWMTrades * getPrice('IWM')
    
def trade_reporting():
    global report
    
    fill_df = util.df([fs.execution for fs in ib.fills()])[["execId", "time", "side", "shares", "avgPrice"]].set_index("execId")
    profit_df = util.df([fs.commissionReport for fs in ib.fills()])[["execId", "realizedPNL"]].set_index("execId")
    report = pd.concat([fill_df, profit_df], axis = 1).set_index("time").loc[session_start:]
    # report = report.groupby(["time", "side"]).agg({"shares":"sum", "avgPrice":"mean", "realizedPNL":"sum"}).reset_index().set_index("time")
    report["cumPNL"] = report.realizedPNL.cumsum()
        
    os.system('clear')
    print(report)
    print(f'Realized P&L of the session: {report.cumPNL[-1]}')
    return report, report.cumPNL[-1]

if __name__ == '__main__':
    featured = pd.read_csv("featured_data.csv", index_col="time", parse_dates=['time'])
    cols = featured.iloc[:, 2:].columns
    test_res, low, high, model, mu, std = fit_model()
    # prepare necessary data
    
    start_session()
    # IBKR TWS streaming and connection