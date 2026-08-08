import streamlit as st
import yfinance as yf
import numpy as np
import scipy.stats as si
import pandas as pd
import matplotlib.pyplot as plt

# --- Black-Scholes Formula ---
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return (S*si.norm.cdf(d1) - K*np.exp(-r*T)*si.norm.cdf(d2))
    else:
        return (K*np.exp(-r*T)*si.norm.cdf(-d2) - S*si.norm.cdf(-d1))

# --- Greeks ---
def greeks(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = si.norm.cdf(d1) if option_type=="call" else -si.norm.cdf(-d1)
    gamma = si.norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*si.norm.pdf(d1)*np.sqrt(T)
    theta = -(S*si.norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*si.norm.cdf(d2 if option_type=="call" else -d2)
    rho = K*T*np.exp(-r*T)*(si.norm.cdf(d2) if option_type=="call" else -si.norm.cdf(-d2))
    return {"Delta":delta,"Gamma":gamma,"Vega":vega,"Theta":theta,"Rho":rho}

# --- Strategy Recommendation ---
def recommend_strategy(IV, HV, outlook="neutral"):
    if IV > HV*1.2:
        return "Iron Condor / Credit Spread (sell volatility)"
    elif IV < HV*0.8:
        return "Long Straddle / Strangle (buy volatility)"
    elif outlook == "bullish":
        return "Covered Call / Bull Call Spread"
    elif outlook == "bearish":
        return "Protective Put / Bear Put Spread"
    else:
        return "Butterfly Spread / Calendar Spread"

# --- Payoff Diagrams + Metrics ---
def payoff_metrics(strategy, S, K, premium=10):
    prices = np.linspace(S*0.7, S*1.3, 100)
    payoff = []

    if "Covered Call" in strategy:
        payoff = [(p-S) - max(0, p-K) + premium for p in prices]
        max_profit = premium + max(0, K-S)
        max_loss = S - premium
        breakeven = S - premium
    elif "Protective Put" in strategy:
        payoff = [(p-S) + max(0, K-p) - premium for p in prices]
        max_profit = (K-S) - premium
        max_loss = premium
        breakeven = S + premium
    elif "Iron Condor" in strategy:
        payoff = [min(max(p-(K-100),0),100) - min(max(p-(K+100),0),100) for p in prices]
        max_profit = 100
        max_loss = 100
        breakeven = (K-100, K+100)
    elif "Straddle" in strategy:
        payoff = [max(0, p-K) + max(0, K-p) - 2*premium for p in prices]
        max_profit = "Unlimited"
        max_loss = 2*premium
        breakeven = (K-premium, K+premium)
    elif "Butterfly" in strategy:
        payoff = [max(0, p-(K-50)) - 2*max(0, p-K) + max(0, p-(K+50)) for p in prices]
        max_profit = 50
        max_loss = premium
        breakeven = (K-50, K+50)
    else:
        payoff = [0 for p in prices]
        max_profit = 0
        max_loss = 0
        breakeven = None

    return prices, payoff, max_profit, max_loss, breakeven

# --- Streamlit UI ---
st.title("Options Strategy Simulator with Outlook & Strikes")

symbol = st.text_input("Enter NSE stock symbol (e.g., RELIANCE, INFY)", "RELIANCE")
strike = st.number_input("Strike Price", value=2500)
days = st.slider("Days to Expiry", 1, 90, 30)

ticker = f"{symbol}.NS"
stock = yf.Ticker(ticker)
S = stock.history(period="1d")['Close'].iloc[-1]
r = 0.06
T = days/365

# Volatility estimates with safe fallback
hist = stock.history(period="1y")
if "Adj Close" in hist.columns:
    data = hist["Adj Close"]
else:
    data = hist["Close"]

log_returns = np.log(data/data.shift(1)).dropna()
HV = np.std(log_returns)*np.sqrt(252)
IV = HV*1.1  # placeholder, replace with NSE chain IV

# --- Market Outlook Prediction ---
trend = data.tail(20).mean() - data.tail(60).mean()
if IV > HV*1.2 and abs(trend) < 0.01*S:
    outlook = "neutral"
elif trend > 0:
    outlook = "bullish"
else:
    outlook = "bearish"

# Strategy recommendation
strategy = recommend_strategy(IV, HV, outlook)

# Suggested strikes
ATM = round(S, -2)  # nearest 100
OTM_call = round(S*1.05, -2)
OTM_put = round(S*0.95, -2)

# Greeks
greek_vals = greeks(S, strike, T, r, IV, "call")
st.subheader("Greeks")
st.write(greek_vals)

st.subheader("Volatility Analysis")
st.write(f"Implied Volatility (IV): {IV:.2f}")
st.write(f"Historical Volatility (HV): {HV:.2f}")

st.subheader("Predicted Market Outlook")
st.write(outlook)

st.subheader("Recommended Strategy")
st.write(strategy)

st.subheader("Suggested Strikes")
st.write(f"ATM Strike: {ATM}")
st.write(f"OTM Call Strike: {OTM_call}")
st.write(f"OTM Put Strike: {OTM_put}")

# Payoff diagram + metrics
prices, payoff, max_profit, max_loss, breakeven = payoff_metrics(strategy, S, strike)
fig, ax = plt.subplots()
ax.plot(prices, payoff, label=strategy)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xlabel("Stock Price at Expiry")
ax.set_ylabel("Profit / Loss")
ax.legend()
st.pyplot(fig)

st.subheader("Strategy Metrics")
st.write(f"Max Profit: {max_profit}")
st.write(f"Max Loss: {max_loss}")
st.write(f"Breakeven: {breakeven}")
