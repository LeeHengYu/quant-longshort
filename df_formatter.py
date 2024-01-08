import numpy as np
import ta

def add_features(df, ticker_list):
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


def extract_features(df, t1: str, t2: str):
    # constant filtering
    features = ['_AroonIndicator', '_EMA_10', '_EMA_20', '_EMA_5', '_EMA_50', '_HBol_10', '_HBol_3', '_HBol_5', '_KD_D', '_KD_K', '_LBol_10', '_LBol_3', '_LBol_5', '_MACD', '_MACD_hist', '_RSI_10', '_RSI_5', '_SAR', '_SMA_50_150']

    cols = []
    for f in features:
        cols.append(t1 + f)
        cols.append(t2 + f)
    return df[cols]