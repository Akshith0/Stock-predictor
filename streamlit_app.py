import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Stock Predictor", layout="wide")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str = "10y") -> pd.DataFrame:
    try:
        df = yf.download(ticker.upper(), period=period, auto_adjust=True, progress=False)
        df = df.dropna()
        # Flatten MultiIndex columns if present (fixes KeyError bug)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Could not load data for {ticker}. Error: {e}")
        return pd.DataFrame()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic_k(df: pd.DataFrame, k_window: int = 14) -> pd.Series:
    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    return 100 * (df['Close'] - low_min) / (high_max - low_min)

def bollinger_width(series: pd.Series, window: int = 20, num_std: int = 2) -> pd.Series:
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return (upper - lower) / ma

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['Return_1d'] = data['Close'].pct_change()
    data['SMA_10'] = data['Close'].rolling(10).mean()
    data['SMA_20'] = data['Close'].rolling(20).mean()
    data['SMA_50'] = data['Close'].rolling(50).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['RSI_14'] = rsi(data['Close'], 14)
    macd_line, signal_line, hist = macd(data['Close'])
    data['MACD'] = macd_line
    data['MACD_Signal'] = signal_line
    data['MACD_Hist'] = hist
    data['Stoch_K'] = stochastic_k(data)
    data['BB_Width'] = bollinger_width(data['Close'])
    data['Vol_Change'] = data['Volume'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    data['Momentum_10'] = data['Close'].pct_change(10)
    data['Momentum_20'] = data['Close'].pct_change(20)
    data = data.dropna()
    return data

def label_target(close: pd.Series, horizon: int = 5, up_th: float = 0.01, down_th: float = -0.01) -> pd.Series:
    future_return = close.shift(-horizon) / close - 1.0
    y = pd.Series(index=close.index, dtype=int)
    # Assign Buy/Sell/Hold labels
    y.loc[future_return > up_th] = 1      # Buy
    y.loc[future_return < down_th] = -1   # Sell
    y.loc[(future_return <= up_th) & (future_return >= down_th)] = 0  # Hold
    return y

def rules_signal(row) -> int:
    buy_votes = 0
    sell_votes = 0

    # Moving average crossover
    if row['SMA_10'] > row['SMA_20'] > row['SMA_50']:
        buy_votes += 1
    if row['SMA_10'] < row['SMA_20'] < row['SMA_50']:
        sell_votes += 1

    # RSI
    if row['RSI_14'] < 30:
        buy_votes += 1
    if row['RSI_14'] > 70:
        sell_votes += 1

    # MACD
    if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
        buy_votes += 1
    if row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
        sell_votes += 1

    if buy_votes > sell_votes:
        return 1
    if sell_votes > buy_votes:
        return -1
    return 0

def recommendation_text(label: int) -> str:
    return {1: "BUY", -1: "SELL", 0: "HOLD"}.get(label, "HOLD")

# ---------- UI ----------
st.title("Any-Ticker Stock Predictor")
st.caption("Enter a stock ticker. The app trains on past data and gives a Buy, Sell, or Hold suggestion. (Educational only, not financial advice.)")

colA, colB, colC = st.columns([2,1,1])
with colA:
    ticker = st.text_input("Ticker", value="AAPL").strip()
with colB:
    horizon = st.selectbox("Prediction horizon (trading days ahead)", [1, 3, 5, 10, 20], index=0)
with colC:
    period = st.selectbox("History window to pull", ["5y", "10y", "max"], index=1)

if st.button("Run model", type="primary") or ticker:
    if not ticker:
        st.warning("Please enter a ticker symbol.")
        st.stop()

    df = load_data(ticker, period=period)
    if df.empty or len(df) < 250:
        st.error("Not enough data to train. Try a different ticker or longer period.")
        st.stop()

    st.subheader(f"Price chart for {ticker.upper()}")
    st.line_chart(df['Close'])

    data = make_features(df)
    y = label_target(data['Close'], horizon=horizon)
    X = data.drop(columns=['Adj Close'], errors='ignore').copy()

    # Align X and y
    X = X.loc[y.index]
    dataset = X.dropna().copy()
    y = y.loc[dataset.index]

    # Remove last 'horizon' rows since their target is NaN
    dataset = dataset.iloc[:-horizon, :]
    y = y.iloc[:-horizon]

    feature_cols = [
        'Return_1d','SMA_10','SMA_20','SMA_50','EMA_10','EMA_20','RSI_14',
        'MACD','MACD_Signal','MACD_Hist','Stoch_K','BB_Width','Vol_Change',
        'Range','Momentum_10','Momentum_20'
    ]
    dataset = dataset[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = y.loc[dataset.index]

    if dataset.empty or y.empty or len(dataset) < 300:
        st.error("Not enough clean rows after feature engineering. Try a different ticker or period.")
        st.stop()

    # Train/test split
    split_idx = int(len(dataset) * 0.8)
    X_train, X_test = dataset.iloc[:split_idx], dataset.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model accuracy (last 20%)", f"{acc*100:.1f}%")
    with col2:
        st.write("Class balance in test set:")
        st.write(y_test.value_counts(normalize=True).rename({-1:"SELL",0:"HOLD",1:"BUY"}).to_frame("Proportion"))

    # Latest prediction
    latest_row = dataset.iloc[[-1]].copy()
    try:
        proba = clf.predict_proba(latest_row)[0]
        class_to_idx = {c:i for i,c in enumerate(clf.classes_)}
        p_sell = proba[class_to_idx.get(-1, 0)]
        p_hold = proba[class_to_idx.get(0, 1)]
        p_buy  = proba[class_to_idx.get(1, 2)]
    except Exception:
        p_sell, p_hold, p_buy = 0.33, 0.34, 0.33

    model_label = int(clf.predict(latest_row)[0])

    # Rules-based vote
    last_feat = data.iloc[-1]
    rule_label = rules_signal(last_feat)

    # Ensemble
    ensemble_score = model_label + rule_label
    if ensemble_score > 0:
        final_label = 1
    elif ensemble_score < 0:
        final_label = -1
    else:
        final_label = 0

    st.subheader("Recommendation")
    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("Model vote", recommendation_text(model_label))
        st.write(pd.DataFrame({"SELL":[p_sell], "HOLD":[p_hold], "BUY":[p_buy]}).T.rename(columns={0:"Probability"}))
    with colB:
        st.metric("Rules vote", recommendation_text(rule_label))
        st.write("Rules consider SMA stack, RSI, and MACD cross.")
    with colC:
        st.metric("Final suggestion", recommendation_text(final_label))

    if final_label == 1:
        st.success("👉 Suggestion: BUY (based on model + rules)")
    elif final_label == -1:
        st.error("👉 Suggestion: SELL (based on model + rules)")
    else:
        st.info("👉 Suggestion: HOLD (based on model + rules)")

    st.subheader("Feature snapshot")
    st.dataframe(latest_row.T.rename(columns={latest_row.index[-1]:"Latest"}))

    st.divider()
    st.caption("This tool is for educational purposes only. It is not financial advice. Do your own research.")
