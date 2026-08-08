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
st.title("Options Strategy Simulator with Greeks Surface - Developed by Sudhagar K")

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

# --- Interactive Greek Selection ---
greek_choice = st.selectbox(
    "Select Greek for 3D Surface",
    ["Delta", "Gamma", "Vega", "Theta", "Rho"]
)

# Define ranges
strike_range = np.arange(S*0.8, S*1.2, 50)   # 20% band around ATM
expiry_range = np.linspace(1/365, 90/365, 30)  # 1 to 90 days

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
