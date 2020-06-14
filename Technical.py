import pandas as pd
import numpy as np
import talib as ta

df = pd.read_csv('BA.csv')

df['Date'] = pd.to_datetime(df['Date'])

df.index = df['Date']

df = df.drop(["Date"], axis=1)


#5.1. Step 1: extract technical indicators

#Define Technical Indicators
#_______________________________________________________________________

#Simple Moving Average  
df['MA'] = ta.SMA(df.Close,100)
df = df.fillna(0)

# RSI
df["RSI"] = ta.RSI(df.Close,14)

# Stochastic K
df["Stoch_K"], df["Stoch_D"] = ta.STOCH(df["High"], df["Low"], df["Close"], fastk_period=14, slowk_period=3, slowk_matype=0, 
                                       slowd_period=3, slowd_matype=0)

# MACD
df['MACD'], df['MACDSignal'], df['MACDHist'] = ta.MACD(df.Close, fastperiod= 12, slowperiod= 26, signalperiod= 9)

# Williams R%
df["WILLR"] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14) 



# 5.2. Step 2: trend analysis using technical indicators

#make MA25 into a series
MA = df.iloc[:,6]

# Up trend is 1, No trend is 0, and Down trend is -1
df['Trend'] = MA.rolling('5d').apply(lambda x: np.sign(x[-1] - x[0]), raw=False)

df.tail()
#If close > MA25 and MA25 is rising for last 5 days then 1 (Uptrend)
#If close < MA25 and MA25 is declining for last 5 days then -1 (Downtrend)
#If neither are true then 0 (Notrend)
def f(row):
    if row['Close'] > row['MA'] and row['Trend'] == 1:
        val = 1
    elif row['Close'] < row['MA'] and row['Trend'] == -1:
        val = 0
    else:
        val = 0
    return val
df['Trend'] = df.apply(f, axis=1, raw=False)


Close = df.iloc[:,3]
df["UpTrend"] = ((Close - Close.rolling('3 D').min())/(Close.rolling('3 D').max() - Close.rolling('3 D').min()) * .5) + .5
df["DownTrend"] = (Close - Close.rolling('3 D').min())/(Close.rolling('3 D').max() - Close.rolling('3 D').min()) * .5

def g(row):
    if row['Trend'] == 1:
        val = row['UpTrend']
    elif row['Trend'] == 0:
        val = row['DownTrend']
    else:
        val = 0
    return val
df['Trade_Signal'] = df.apply(g, axis=1, raw=False)


#I added .tail(n=Length - 32) to omit values where the indicators = 0 since the machine learning agorithm trains on the values 
#of the indicators and the MACD doesn't generate values until 33 periods have passed.
Length = (df.shape[0] - 32)

df2 = (df[['MA', 'RSI', 'Stoch_K', 'Stoch_D', 'MACD', 'WILLR', 'Trade_Signal']].copy().tail(n=Length).copy())

df2 = df2.fillna(0)


#5.5. Step 5: network structure creation and training using ELM

target_column = ['Trade_Signal'] 
predictors = list(set(list(df2.columns))-set(target_column))
#line 3 here fufills the data normalization
df2[predictors] = (df2[predictors] - df2[predictors].min())/(df2[predictors].max() - df2[predictors].min())

train = df2.iloc[:1000, :]
test = df2.iloc[1000:, :]

X= train[predictors]

y = train[target_column]


synaptic_weights = np.random.uniform(low=0, high=1, size=(6,1))
#insert constant weight 1 (W0)
synaptic_weights = np.append(synaptic_weights, 1)


#Expand inputs with functional expansion block
feb = np.outer(X, synaptic_weights)
feb2 = np.sum(feb.reshape(-1, 7), axis=1)
expanded_inputs = np.tanh(feb2).reshape(len(X), -1)
