import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

st.set_page_config(page_title="WGIC Stock Simulator", layout="wide")
st.title("📈 Stock Predictor & Monte Carlo Simulation")

# ---------- Ticker selection ----------
ticker_list = [
    "AAL","AAPL","ACHC","ADBE","AEHR","AEP","AMD","AMGN","AMTX","AMZN","ARCB","AVGO",
    "BECN","BIDU","CAAS","CAKE","CASY","CHNR","CHPT","CMCSA","COST","CPRX","CSCO","CTSH",
    "CZR","DBX","DJCO","DLTR","ETSY","FIZZ","FTNT","GBCI","GEG","GILD","GMAB","GOGO",
    "GOOGL","GRPN","HAS","HBIO","HTLD","ILMN","INTC","IOSP","JBLU","KALU","KDP","LE","LQDA",
    "LULU","LYFT","MANH","MAR","MAT","META","MIDD","MNST","MSEX","MSFT","MTCH","MYGN","NCTY",
    "NTES","NTIC","NVDA","NXPI","ONB","ORLY","OZK","PCAR","PEP","PTON","PYPL","PZZA","QCOM",
    "REGN","RGLD","ROCK","RTC","SBUX","SEDG","SEIC","SFIX","SFM","SIRI","SKYW","SOHU","SWBI",
    "TROW","TSLA","TXN","TXRH","ULTA","URBN","USLM","UTSI","VEON","VRA","VRSK","WBA","WDFC",
    "WEN","YORW","ABBV","ABT","AEO","AFL","ALL","AMC","AMN","ANET","ANF","APAM","APD","APTV",
    "ASGN","ASH","AWK","AXP","AZO","BA","BABA","BAC","BAM","BAX","BBW","BBY","BCS","BEN","BILL",
    "BLK","BMY","BNED","BP","BUD","BURL","BWA","BX","C","CAT","CCJ","CL","CLW","CMG","CNC",
    "CNI","CP","CPB","CRH","CRM","CTVA","CVS","CVX","CYD","D","DAL","DB","DE","DEO","DFS","DG",
    "DIS","DLR","DOC","DOW","DXC","EDR","EDU","EL","EMN","ENB","ET","EXR","F","FCN","FCX","FE",
    "FICO","FL","FMC","FTS","GD","GE","GEO","GIS","GM","GMED","GRMN","GS","GSK","H","HD","HES",
    "HMC","HOG","HRB","HSY","ICE","IMAX","IQV","IRM","JNJ","JPM","K","KEY","KKR","KMI","KMX",
    "KO","KWR","L","LAC","LAZ","LCII","LMT","LOW","LUV","LVS","M","MA","MCD","MCK","MCO","MET",
    "MKC","MOV","MRK","MS","MTB","NCLH","NFG","NGS","NKE","NOC","NOV","NTR","NVO","NVS","OKE",
    "OPY","ORCL","PBH","PCG","PFE","PG","PKX","PLNT","PLOW","PNC","PRU","PSA","PSX","RBA",
    "RCI","RF","RTX","SAP","SAVE","SCHW","SJW","SNA","SNOW","SO","SONY","SPOT","SRE","SUN",
    "SYY","T","TAL","TAP","TCS","TEVA","TGT","THS","TJX","TM","TR","TREX","TRP","TSM","TSN",
    "TU","TWI","TXT","UA","UBER","UBS","UGI","UL","UNFI","UNH","UPS","V","VEEV","VFC","VZ",
    "WFC","WH","WHD","WMT","WNC","WSM","X","XOM","XRX","YUM","ZTO"
]

ticker = st.selectbox("Select stock ticker:", ticker_list)

# ---------- Monte Carlo settings ----------
num_simulations = st.slider("Number of simulations:", 1000, 50000, 10000, 1000)
horizon_days = st.number_input("Prediction horizon (days ahead):", min_value=1, max_value=365, value=30)

if st.button("Run Prediction"):

    # ---------- Fetch historical data ----------
    end_date = date.today()
    start_date = end_date - timedelta(days=365*2)  # 2 years history
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        st.error("No data found. Check ticker.")
        st.stop()
    
    # ---------- SMA signals ----------
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["Signal"] = 0
    df.loc[df.index[19]:, "Signal"] = (df["SMA20"].iloc[19:] > df["SMA50"].iloc[19:]).astype(int)
    df["Position"] = df["Signal"].diff()

    # Plot price + signals
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df["Close"], label="Close Price", color="blue")
    ax.plot(df.index, df["SMA20"], label="SMA20", color="green")
    ax.plot(df.index, df["SMA50"], label="SMA50", color="red")
    ax.scatter(df[df["Position"]==1].index, df["SMA20"][df["Position"]==1], marker="^", color="g", s=100, label="Buy Signal")
    ax.scatter(df[df["Position"]==-1].index, df["SMA20"][df["Position"]==-1], marker="v", color="r", s=100, label="Sell Signal")
    ax.set_title(f"{ticker} Price with SMA Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    st.pyplot(fig)

    # Current SMA recommendation
    latest_signal = df["Signal"].iloc[-1]
    if latest_signal == 1:
        st.success("✅ SMA Recommendation: BUY")
    else:
        st.info("ℹ️ SMA Recommendation: SELL / HOLD")

    # ---------- Monte Carlo Simulation ----------
    returns = df["Close"].pct_change().dropna()
    mu = float(returns.mean())
    sigma = float(returns.std())
    S0 = float(df["Close"].iloc[-1])

    simulations = np.zeros((horizon_days, num_simulations))
    simulations[0, :] = S0 * (1 + np.random.normal(mu, sigma, num_simulations))

    for t in range(1, horizon_days):
        simulations[t, :] = simulations[t-1, :] * (1 + np.random.normal(mu, sigma, num_simulations))

    final_prices = simulations[-1, :]
    prob_up = (final_prices > S0).mean()
    avg_return = (final_prices / S0 - 1).mean()

    if prob_up > 0.55:
        mc_recommendation = "BUY"
    elif prob_up < 0.45:
        mc_recommendation = "SELL"
    else:
        mc_recommendation = "HOLD"

    # ---------- Plot simplified Monte Carlo ----------
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(simulations[:, :500], color="lightblue", alpha=0.1)
    ax2.set_title(f"Monte Carlo Simulation ({num_simulations} sims, {horizon_days} days ahead)")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price ($)")
    st.pyplot(fig2)

    st.subheader("Monte Carlo Results")
    st.metric("Recommendation", mc_recommendation)
    st.write(f"Chance of going up: {prob_up*100:.2f}%")
    st.write(f"Expected average return: {avg_return*100:.2f}%")
