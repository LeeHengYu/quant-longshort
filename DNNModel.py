import random
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers.legacy import Adam
from keras.regularizers import l1, l2


def set_seeds(seed = 100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
def cw(df): # count weight (only 0 or 1)
    c0, c1 = np.bincount(df["LSportfolio_direction"])
    w0 = (1/c0) * (len(df)) / 2
    w1 = (1/c1) * (len(df)) / 2
    return {0:w0, 1:w1}

# adam = Adam(learning_rate = 0.002)

def create_model(hl = 3, hu = 50, rate = 0.4,
                 reg = l1(0.0005), adam_LR = 0.001, input_dim = None):
    # hl: hidden layers
    # hu: no. of neurons in each hidden layer
    # input layer
    model = Sequential()
    model.add(Dense(hu, input_dim = input_dim, activity_regularizer = reg, activation = "relu"))
    model.add(Dropout(rate, seed = 100))

    # hidden layers
    for _ in range(hl):
        model.add(Dense(hu, activation = "relu", activity_regularizer = reg))
        model.add(Dropout(rate, seed = 100))
    
    # model.add(Dense(hu, activation = "relu", activity_regularizer = reg))
    # model.add(Dropout(rate, seed = 100))
    
    # output layers
    model.add(Dense(1, activation = "sigmoid")) # output layer
    model.compile(loss = "binary_crossentropy", optimizer = Adam(learning_rate = adam_LR), metrics = ["accuracy"])
    # output only has 2 outcome (up or down), so the loss function takes binary_crossentropy
    return model

def add_portfolio_setting(
    df,
    ticker_list = ['QQQ', 'IWM'],
    QQQ_weight = None,
    IWM_weight = None):
    """
    Enter either QQQ or IWM weight
    QQQ_weight: portion of QQQ (usually short side)
    IWM_weight: portion of IWM (usually long side)
    QQQ_weight + IWM
    
    DataFrame is modified in place
    """
    if QQQ_weight == None:
        QQQ_weight = 1 - IWM_weight
    if IWM_weight == None:
        IWM_weight = 1 - QQQ_weight
        
    if not QQQ_weight and not IWM_weight:
        raise Exception("Invalid weightage.")
    
    LSportfolio_return = df[ticker_list[0] + '_return'] * QQQ_weight + df[ticker_list[1] + '_return'] * IWM_weight
    df['LSportfolio_direction'] = np.where(LSportfolio_return > 0, 1, 0)
    
def try_params(comb, rf = 0.04, interruption = None):
    global test_result
    
    base_feature = pd.read_csv("featured_data.csv", index_col="time")
    cols = base_feature.iloc[:, 2:].columns
    test_result = defaultdict(list) # empty test_result
    for wQQQ, LR, epoch, neuron, layer in comb:
        if interruption and len(test_result['Sharpe']) == interruption: break
        
        featured = base_feature.copy()
        add_portfolio_setting(featured, QQQ_weight = wQQQ)

        split_idx = int(len(featured) * 0.66667)

        train = featured.iloc[:split_idx]
        test = featured.iloc[split_idx:]

        mu, std = train.mean(), train.std()
        train_s = (train - mu) / std
        featured_s = (featured - mu) / std

        model = create_model(hl = layer, hu = 64, input_dim = len(cols), adam_LR=LR)
        model.fit(x = train_s[cols], y = train["LSportfolio_direction"], epochs = epoch, verbose = False,
          validation_split = 0.25, shuffle = False, class_weight = cw(featured))

        test_s = (test - mu) / std
        accuracy = model.evaluate(x = test_s[cols], y = test["LSportfolio_direction"])[1]


        # append results
        test_result['QQQ_weight'].append(wQQQ)
        test_result['Adam_LR'].append(LR)
        test_result['Epochs'].append(epoch)
        test_result['Neurons'].append(neuron)
        test_result['HiddenLayers'].append(layer)
        test_result['Accuracy'].append(accuracy)

        pred = model.predict(test_s[cols])
        low, high = np.percentile(pred, 20), np.percentile(pred, 80)
        
        # backtesting the strategy return
        bt = test_s.copy()
        bt["pred"] = pred
        bt["position"] = np.where(bt.pred <= low, -1, np.nan)
        bt["position"] = np.where(bt.pred >= high, 1, bt.position)
        bt["position"] = bt.position.fillna(0)
        bt["DNNreturn"] = bt.position.shift(1) * (wQQQ * bt.QQQ_return.apply(np.exp).sub(1) + (1 - wQQQ) * bt.IWM_return.apply(np.exp).sub(1))
        bt["DNNreturn"] = bt.DNNreturn.add(1).apply(np.log)

        # calculate sharpe ratio
        # sum over len(bt)/252 year(s) of log return
        cur_return = np.exp(bt['DNNreturn'].mean() * 252) - 1
        cur_std = bt['DNNreturn'].std() * np.sqrt(252 * 7.5)
        sharpe = (cur_return - rf) / cur_std
        test_result['Strategy_return'].append(cur_return)
        test_result['Strategy_std'].append(cur_std)
        test_result['Sharpe'].append(sharpe)
    return pd.DataFrame(test_result)