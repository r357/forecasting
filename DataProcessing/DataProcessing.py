# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def Splitter(seq_len, data):
  X_data, Y_data = [], []
  for i in range(seq_len, len(data)):
    X_data.append(data[i-seq_len:i]) #Chunks of 128 len
    Y_data.append(data[:, 3][i])
  return(np.array(X_data),np.array(Y_data))



def PrepData(ticker, seq_len):
  ''' 
  Downloads and normalizes (minmax) ticker
  60/20/20 split
  2000-1-1 start
  '''

  # Load Data
  d = pd.DataFrame(yf.download(ticker, start="2000-1-1", end="2021-1-10"))
  
  # Use Adj. Close
  d["Close"] = d["Adj Close"]
  d.drop(columns=["Adj Close"], inplace=True)
  d.dropna()
  df = d.pct_change().dropna()

  #normalize
  retmin = min(df[["Open", "High", "Low", "Close"]].min())
  retmax = max(df[["Open", "High", "Low", "Close"]].max())
  norm = retmax - retmin
  dfn = df.copy()

  for c in df:
      if c != "Volume":
          dfn[c] = (df[c]-retmin)/norm
      else:  
          dfn[c] = (df[c]-df[c].min())/(df[c].max()-df[c].min())

  dfn.reset_index(drop=True, inplace=True)
  train, valid, test = (np.split(dfn, [int(.6*len(dfn)), int(.8*len(dfn))]))
  
  X_train, Y_train = Splitter(seq_len, train.values)
  X_val, Y_val = Splitter(seq_len, valid.values)
  X_test, Y_test = Splitter(seq_len, test.values)

  return (X_train, Y_train, X_val, Y_val, X_test, Y_test)


