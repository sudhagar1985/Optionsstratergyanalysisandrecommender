import streamlit as st
import yfinance as yf
import numpy as np
import scipy.stats as si
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# --- Payoff Metrics ---
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

    elif "Bear Put Spread" in strategy:
        lower = K-100
        payoff = [max(0, K-p) - max(0, lower-p) - premium for p in prices]
        max_profit = (K-lower) - premium
        max_loss = premium
        breakeven = K - premium

    elif "Bull Ratio Spread" in strategy:
        payoff = [max(0, p-K) - 2*max(0, p-(K+100)) for p in prices]
        max_profit = 100
        max_loss = "Unlimited"
        breakeven = (K+100, "Unlimited")

    elif "Bear Ratio Spread" in strategy:
        payoff = [max(0, K-p) - 2*max(0, (K-100)-p) for p in prices]
        max_profit = 100
        max_loss = "Unlimited"
        breakeven = (K-100, "Unlimited")

    else:
        payoff = [0 for p in prices]
        max_profit = 0
        max_loss = 0
        breakeven = None

    return prices, payoff, max_profit, max_loss, breakeven

# --- Best Strikes ---
def best_strikes(strategy, S, strike_range, T, r, sigma, lot=505):
    results = []
    for K in strike_range:
        prices, payoff, max_profit, max_loss, breakeven = payoff_metrics(strategy, S, K)
        results.append({
            "Strike": K,
            "MaxProfit": max_profit,
            "MaxLoss": max_loss,
            "Breakeven": breakeven
        })
    df = pd.DataFrame(results)
    df_numeric = df[df["MaxProfit"].apply(lambda x: isinstance(x,(int,float)))]
    df_top5 = df_numeric.sort_values("MaxProfit", ascending=False).head(5)
    return df_top5

# --- Combined Payoff Chart ---
def combined_payoff_chart(strategy, S, top5_strikes, T, r, sigma):
    fig, ax = plt.subplots()
    for K in top5_strikes["Strike"]:
        prices, payoff, _, _, _ = payoff_metrics(strategy, S, K)
        ax.plot(prices, payoff, label=f"Strike {K}")
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Stock Price at Expiry")
    ax.set_ylabel("Profit / Loss")
    ax.set_title(f"Combined Payoff Chart - {strategy}")
    ax.legend()
    return fig

# --- Greeks Surface ---
def greeks_surface(S, strikes, expiries, r, sigma, option_type="call", greek="Delta"):
    surface = []
    for K in strikes:
        row = []
        for T in expiries:
            g = greeks(S, K, T, r, sigma, option_type)
            row.append(g[greek])
        surface.append(row)
    return np.array(surface)

# --- Streamlit UI ---
st.title("Options Strategy Simulator with Top 5 Strikes & Ratio Spreads - Developed by Sudhagar K")

symbol = st.text_input("Enter NSE symbol (e.g., RELIANCE)", "RELIANCE")
strike = st.number_input("Strike Price", value=2500)
days = st.slider("Days to Expiry", 1, 90, 30)

ticker = f"{symbol}.NS"
stock = yf.Ticker(ticker)
S = stock.history(period="1d")['Close'].iloc[-1]
r = 0.06
T = days/365

# Volatility estimates
hist = stock.history(period="1y")
data = hist["Close"]
log_returns = np.log(data/data.shift(1)).dropna()
HV = np.std(log_returns)*np.sqrt(252)
IV = HV*1.1  # placeholder IV

st.subheader("Market Data")
st.write(f"Spot Price: {S:.2f}")
st.write(f"Implied Volatility (IV): {IV:.2f}")
st.write(f"Historical Volatility (HV): {HV:.2f}")

# --- Strategy Selection ---
strategy = st.selectbox(
    "Select Strategy",
    ["Covered Call", "Protective Put", "Bear Put Spread", "Bull Ratio Spread", "Bear Ratio Spread"]
)

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

# --- Top 5 Strikes ---
strike_range = np.arange(S*0.8, S*1.2, 50)
df_top5 = best_strikes(strategy, S, strike_range, T, r, IV)

st.subheader("Top 5 Strikes for Selected Strategy")
st.write(df_top5)

best_row = df_top5.iloc[0]
st.write("Best Strike:", best_row["Strike"])
st.write("Max Profit:", best_row["MaxProfit"])
st.write("Max Loss:", best_row["MaxLoss"])
st.write("Breakeven:", best_row["Breakeven"])

# --- Combined Payoff Chart ---
st.subheader("Combined Payoff Chart for Top 5 Strikes")
fig = combined_payoff_chart(strategy, S, df_top5, T, r, IV)
st.pyplot(fig)

# --- Interactive Greek Selection ---
greek_choice = st.selectbox(
    "Select Greek for 3D Surface",
    ["Delta", "Gamma", "Vega", "Theta", "Rho"]
)

# Define ranges
strike_range = np.arange(S*0.8, S*1.2, 50)
expiry_range = np.linspace(1/365, 90/365, 30)

# Compute surface for chosen Greek
Z = greeks_surface(S, strike_range, expiry_range, r, IV, "call", greek=greek_choice)
X, Y = np.meshgrid(strike_range, expiry_range)

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y*365, Z.T, cmap="viridis", alpha=0.8)

ax.set_xlabel("Strike Price")
ax.set_ylabel("Days to Expiry")
ax.set_zlabel(f"{greek_choice} Value")
ax.set_title(f"{greek_choice} Surface vs Strike & Expiry")
fig.colorbar(surf, shrink=0.5, aspect=5)

st.pyplot(fig)
