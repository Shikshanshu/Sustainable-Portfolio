#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
import datetime
import calendar

from pypfopt.expected_returns import ema_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import cvxpy as cp

import plotly.express as px
import plotly.graph_objects as go

import urllib.request
import json


# In[2]:


from scipy.stats import norm
from pandas_datareader import data as pdr
import yfinance as yf
yf.pdr_override()


# In[3]:


tickers=['COALINDIA.NS','BPCL.NS','RELIANCE.NS','IOC.NS','ONGC.NS','CIPLA.NS','DRREDDY.NS','SUNPHARMA.NS','DIVISLAB.NS','ITC.NS','HINDUNILVR.NS','NESTLEIND.NS','BRITANNIA.NS','TATACONSUM.NS','LT.NS','ADANIPORTS.NS','SBIN.NS','AXISBANK.NS','KOTAKBANK.NS','INDUSINDBK.NS','ICICIBANK.NS','HDFCBANK.NS','HDFC.NS','SBILIFE.NS','HDFCLIFE.NS','BAJAJFINSV.NS','BAJFINANCE.NS','MARUTI.NS','TATAMOTORS.NS','M&M.NS','EICHERMOT.NS','HEROMOTOCO.NS','BAJAJ-AUTO.NS','TITAN.NS','NTPC.NS','POWERGRID.NS','INFY.NS','WIPRO.NS','HCLTECH.NS','TCS.NS','TECHM.NS','HINDALCO.NS','GRASIM.NS','ASIANPAINT.NS','JSWSTEEL.NS','ULTRACEMCO.NS','UPL.NS','SHREECEM.NS','TATASTEEL.NS','BHARTIARTL.NS']


# In[4]:


esg_list = []
e_list = []
s_list = []
g_list = []
missing = []

for ticker in tickers:
    # Try fetching ESG data from yahoo finance
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/esgChart?symbol={}".format(ticker)

        connection = urllib.request.urlopen(url)

        data_connection = connection.read()
        data_json = json.loads(data_connection)
        formatdata = data_json["esgChart"]["result"][0]["symbolSeries"]
        df_data = pd.DataFrame(formatdata)
        df_data["timestamp"] = pd.to_datetime(df_data["timestamp"], unit="s")
        df_data = df_data.set_index('timestamp')
        df_data = df_data.loc['2012':'2022']
        esg_list.append(df_data['esgScore'])
        e_list.append(df_data['environmentScore'])
        s_list.append(df_data['socialScore'])
        g_list.append(df_data['governanceScore'])
    except:
        missing.append(ticker)
        continue

# If ticker is missing data remove ticker from investment sample universe
if missing != None:
    for tick in missing:
       tickers.remove(tick)

esg_df = pd.concat(esg_list, axis=1)
esg_df.columns = tickers

e_df = pd.concat(e_list, axis=1)
e_df.columns = tickers

s_df = pd.concat(s_list, axis=1)
s_df.columns = tickers

g_df = pd.concat(g_list, axis=1)
g_df.columns = tickers


# In[5]:


esg_df.mean().plot(kind='hist')


# In[6]:


df = pd.DataFrame()
for ticker in tickers:
    df[ticker]=pdr.DataReader(ticker,source='yahoo',start='2013-1-1')['Adj Close']
df


# In[7]:


esg_tot_sample = esg_df[tickers].copy()
esg_e_sample = e_df[tickers].copy()
esg_s_sample = s_df[tickers].copy()
esg_g_sample = g_df[tickers].copy()


# In[8]:


class efficient_frontier():
    def __init__(self, stocks, up_bound, samples, ticks):
        self.stocks = stocks
        self.upper_bound = up_bound
        self.samples = samples
        self.ticks = ticks
    
    # Optimization
    def optimize_portfolios(self):
        # Compute exponentially weighted historical mean returns
        mu = ema_historical_return(self.stocks)
        # Compute covariance matrix using ledoit-wolf shrinkage 
        s = CovarianceShrinkage(self.stocks).ledoit_wolf()
        n = len(mu)
        # Set upper bound for weights
        upper_bound = self.upper_bound
        w = cp.Variable(n)
        gamma = cp.Parameter(nonneg=True)
        ret = mu.to_numpy().T@w 
        risk = cp.quad_form(w, s.to_numpy())
        # Utility function to optimize
        prob = cp.Problem(cp.Maximize(ret - gamma*risk), 
                          [cp.sum(w) == 1, w >= 0, w <= upper_bound])
        
        risk_data = np.zeros(self.samples)
        ret_data = np.zeros(self.samples)
        weights_data = []
        # Gamma variables for utility function
        gamma_vals = np.logspace(-2, 3, num=self.samples)

        for i in range(self.samples):
            gamma.value = gamma_vals[i]
            prob.solve()
            weights = w
            weights_tick = pd.Series(weights.value, index=self.ticks)
            weights_tick_filter = weights_tick[weights_tick > 1.0e-03]
            weights_data.append(weights_tick_filter)
            risk_data[i] = cp.sqrt(risk).value
            ret_data[i] = ret.value
        # DataFrame with weights of stocks    
        weights_df = pd.concat(weights_data, axis = 1).T.fillna(0)
        # DataFrame with returns and risk
        rr_df = pd.DataFrame({'Return': ret_data, 'Risk': risk_data})
        self.portfolios_df = pd.concat([rr_df, weights_df], axis = 1)


# In[9]:


conv_frontier = efficient_frontier(df, 0.05, 200, tickers)
conv_frontier.optimize_portfolios()


# In[10]:


def screen(esg_df, stocks_df):
    mean_esg = esg_df.mean()
    thresh = mean_esg.quantile(0.3)
    mean_esg = mean_esg[mean_esg > thresh]
    screen_stocks = stocks_df[mean_esg.index]
    screen_ticks = screen_stocks.columns
    return screen_stocks, screen_ticks


# In[11]:


e_stocks, e_ticks = screen(esg_e_sample, df)
e_frontier = efficient_frontier(e_stocks, 0.05, 200, e_ticks)
e_frontier.optimize_portfolios()

# S Screen
s_stocks, s_ticks = screen(esg_s_sample, df)
s_frontier = efficient_frontier(s_stocks, 0.05, 200, s_ticks)
s_frontier.optimize_portfolios()

# G Screen
g_stocks, g_ticks = screen(esg_g_sample, df)
g_frontier = efficient_frontier(g_stocks, 0.05, 200, g_ticks)
g_frontier.optimize_portfolios()

# ESG Screen
esg_stocks, esg_ticks = screen(esg_tot_sample, df)
esg_frontier = efficient_frontier(esg_stocks, 0.05, 200, esg_ticks)
esg_frontier.optimize_portfolios()


# In[12]:


rr_conv = conv_frontier.portfolios_df[['Return', 'Risk']].copy()
rr_conv['Screen'] = 'No Screen'

rr_e = e_frontier.portfolios_df[['Return', 'Risk']].copy()
rr_e['Screen'] = 'E'

rr_s = s_frontier.portfolios_df[['Return', 'Risk']].copy()
rr_s['Screen'] = 'S'

rr_g = g_frontier.portfolios_df[['Return', 'Risk']].copy()
rr_g['Screen'] = 'G'

rr_esg = esg_frontier.portfolios_df[['Return', 'Risk']].copy()
rr_esg['Screen'] = 'ESG'

rr_all = pd.concat([rr_conv, rr_e, rr_s, rr_g, rr_esg])


# In[13]:


px.scatter(rr_all, x='Risk', y='Return', color='Screen', width=1000, height=700)


# In[14]:


def get_sharpe_rr_and_weights(df):
    df['Sharpe'] = (df['Return'] - 0.02) / df['Risk']
    sharpe_rr = df.iloc[df['Sharpe'].idxmax()].loc[['Return', 'Risk']]
    sharpe_weights = df.drop(['Return', 'Risk', 'Sharpe'], axis = 1).iloc[df['Sharpe'].idxmax()]
    sharpe_weights = sharpe_weights[sharpe_weights!=0]
    return sharpe_rr, sharpe_weights


# In[15]:


conv_sh_rr, conv_sh_w = get_sharpe_rr_and_weights(conv_frontier.portfolios_df)
e_sh_rr, e_sh_w = get_sharpe_rr_and_weights(e_frontier.portfolios_df)
s_sh_rr, s_sh_w = get_sharpe_rr_and_weights(s_frontier.portfolios_df)
g_sh_rr, g_sh_w = get_sharpe_rr_and_weights(g_frontier.portfolios_df)
esg_sh_rr, esg_sh_w = get_sharpe_rr_and_weights(esg_frontier.portfolios_df)


# In[16]:


sh_rr_all = pd.concat([conv_sh_rr, e_sh_rr, s_sh_rr, g_sh_rr, esg_sh_rr], axis=1).T
sh_rr_all['Screen'] = ['No Screen', 'E', 'G', 'S', 'ESG']
sh_rr_all


# In[17]:


fig = px.line(rr_all, x='Risk', y='Return', color='Screen')
fig.add_trace(
    go.Scatter(
        x=sh_rr_all['Risk'],
        y=sh_rr_all['Return'],
        mode='markers',
        marker=dict(
            color='LightSkyBlue',
            size=8,
            line=dict(
                color='MediumPurple',
                width=2
            )
        ),
        marker_symbol='cross-dot',
        showlegend=False,
    )
)
fig.update_layout(
    width=1000,
    height=700,
)


# In[18]:


def compute_sh_portfolio(df, ticks, weights):
    stock_rets = df[ticks].pct_change().fillna(0)
    weighted_stock_rets = stock_rets * weights
    port_rets = weighted_stock_rets.sum(axis=1)
    cum_port_rets = (port_rets + 1).cumprod()
    return port_rets, cum_port_rets


# In[19]:


conv_sh_rets, conv_sh_port = compute_sh_portfolio(df, conv_sh_w.index, conv_sh_w.values)

e_sh_rets, e_sh_port = compute_sh_portfolio(df, e_sh_w.index, e_sh_w.values)

g_sh_rets, g_sh_port = compute_sh_portfolio(df, g_sh_w.index, g_sh_w.values)

s_sh_rets, s_sh_port = compute_sh_portfolio(df, s_sh_w.index, s_sh_w.values)

esg_sh_rets, esg_sh_port = compute_sh_portfolio(df, esg_sh_w.index, esg_sh_w.values)


# In[20]:


all_sh_port = pd.concat([conv_sh_port,e_sh_port, s_sh_port, g_sh_port, esg_sh_port], axis = 1)
all_sh_port.columns = ['No Screen', 'E', 'S', 'G', 'ESG']
all_sh_port.head()


# In[21]:


fig = px.line(all_sh_port, x=all_sh_port.index, y=all_sh_port.columns)
fig.update_layout(
    width=1000,
    height=700,
)


# In[22]:


(all_sh_port.iloc[-1] - all_sh_port.iloc[0]) / all_sh_port.iloc[0]


# In[ ]:




