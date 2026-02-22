import streamlit as st
import pandas as pd
import json
import os
import yfinance as yf
import plotly.graph_objects as go

from stock_engine import analyze_trading_signals
from data.sp500_list import TICKERS as SP500_TICKERS
from data.nasdaq100_list import TICKERS as NASDAQ100_TICKERS

TRACK_FILE = "data/tracked_stocks.json"

# ---------------------------------------
# HELPERS
# ---------------------------------------
@st.cache_data
def get_company_name(ticker):
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName", "")
    except:
        return ""

def load_tracked_stocks():
    if not os.path.exists(TRACK_FILE):
        return []
    try:
        with open(TRACK_FILE, "r") as f:
            data = json.load(f)
            return data.get("tracked", [])
    except:
        return []

def save_tracked_stocks(tracked_list):
    with open(TRACK_FILE, "w") as f:
        json.dump({"tracked": tracked_list}, f, indent=4)

@st.cache_data
def fetch_history(ticker, period="180d"):
    try:
        df = yf.download(ticker, period=period)
        if df is None or df.empty:
            return None
        return df
    except:
        return None

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ---------------------------------------
# BULLETPROOF SPC CHART
# ---------------------------------------
def plot_spc_chart(ticker, window, control_k):

    hist = fetch_history(ticker, period="180d")
    if hist is None or hist.empty:
        hist = fetch_history(ticker, period="1y")

    if hist is None or hist.empty:
        st.warning("No valid price history available for this ticker.")
        return

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    if "Close" not in hist.columns:
        st.warning("Price history is missing a 'Close' column for this ticker.")
        return

    if hist["Close"].dropna().empty:
        st.warning("All Close prices are NaN — cannot plot SPC chart.")
        return

    hist = hist.dropna(subset=["Close"])

    if len(hist) < window:
        st.warning(f"Not enough data to compute MA{window}.")
        return

    close = hist["Close"]
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    ucl = ma + control_k * std
    lcl = ma - control_k * std
    rsi = compute_rsi(close)

    df_plot = pd.DataFrame({
        "Close": close,
        "MA": ma,
        "UCL": ucl,
        "LCL": lcl,
        "RSI": rsi
    }).dropna()

    if df_plot.empty:
        st.warning("Not enough valid data to plot SPC chart.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["Close"], name="Price", line=dict(color="white")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["MA"], name=f"MA{window}", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["UCL"], name="Upper Control Limit", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["LCL"], name="Lower Control Limit", line=dict(color="green", dash="dash")))

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot["RSI"], name="RSI", line=dict(color="cyan"), yaxis="y2"))

    fig.update_layout(
        height=700,
        template="plotly_dark",
        yaxis=dict(domain=[0.35, 1.0], title="Price"),
        yaxis2=dict(domain=[0.0, 0.25], title="RSI", anchor="x"),
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="HopTrading",
    layout="wide"
)

st.title("📉 Hoptranding")

st.write(
    "This page analyzes short‑term trading opportunities using SPC‑style control limits, "
    "drift direction, RSI, and band width to identify predictable oscillation patterns. "
    "It also maintains a dynamic list of **Stocks Being Tracked** to support a rotating portfolio."
)

# ---------------------------------------
# LOAD TRACKED STOCKS
# ---------------------------------------
tracked = load_tracked_stocks()

# ---------------------------------------
# SIDEBAR — MANAGE TRACKED STOCKS
# ---------------------------------------
st.sidebar.header("📌 Manage Tracked Stocks")

selected_tracked = st.sidebar.multiselect(
    "Tracked Stocks",
    options=tracked,
    default=tracked,
    format_func=lambda x: f"{x} — {get_company_name(x)}"
)

new_ticker = st.sidebar.text_input("Add a ticker (e.g., AAPL)").upper()

if st.sidebar.button("Add Ticker"):
    if new_ticker and new_ticker not in tracked:
        tracked.append(new_ticker)
        save_tracked_stocks(sorted(list(set(tracked))))
        st.sidebar.success(f"Added {new_ticker} — {get_company_name(new_ticker)}")
    else:
        st.sidebar.warning("Ticker is empty or already tracked.")

if st.sidebar.button("Remove Selected"):
    tracked = [t for t in tracked if t not in selected_tracked]
    save_tracked_stocks(tracked)
    st.sidebar.success("Selected tickers removed.")

if st.sidebar.button("Reset Tracked List"):
    tracked = []
    save_tracked_stocks([])
    st.sidebar.success("Tracked list reset.")

if st.sidebar.button("Save Changes"):
    save_tracked_stocks(sorted(list(set(selected_tracked))))
    st.sidebar.success("Tracked list saved.")

tracked = load_tracked_stocks()

# ---------------------------------------
# DISPLAY TRACKED STOCKS
# ---------------------------------------
st.subheader("📌 Stocks Being Tracked")

if tracked:
    tracked_display = [f"{t} — {get_company_name(t)}" for t in tracked]
    st.write(", ".join(tracked_display))
else:
    st.info("No stocks are being tracked yet. Using selected universe as starting point.")

# ---------------------------------------
# SIDEBAR — PARAMETERS
# ---------------------------------------
st.sidebar.header("Analysis Parameters")

window = st.sidebar.number_input("Control Window (days)", 5, 60, 20)
control_k = st.sidebar.number_input("Control Limit Multiplier (k)", 1.0, 4.0, 2.0)
max_hold_days = st.sidebar.number_input("Max Hold Window (days)", 3, 30, 10)

# ---------------------------------------
# UNIVERSE SELECTION
# ---------------------------------------
st.subheader("Select Universe")

universe_choice = st.selectbox(
    "Choose a universe",
    ["S&P 500", "NASDAQ 100", "Custom"]
)

if universe_choice == "S&P 500":
    universe = SP500_TICKERS
elif universe_choice == "NASDAQ 100":
    universe = NASDAQ100_TICKERS
else:
    custom_input = st.text_input(
        "Enter tickers separated by commas",
        value="AAPL, MSFT, NVDA"
    )
    universe = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

st.write(f"Analyzing **{len(universe)}** symbols…")

# ---------------------------------------
# RUN ANALYSIS
# ---------------------------------------
results = []

progress = st.progress(0.0)
for i, symbol in enumerate(universe):
    res = analyze_trading_signals(
        symbol,
        window=window,
        control_k=control_k,
        max_hold_days=max_hold_days
    )
    if res:
        results.append(res)

    progress.progress((i + 1) / len(universe))

if not results:
    st.warning("No results available.")
    st.stop()

# ---------------------------------------
# BUILD TABLE
# ---------------------------------------
df = pd.DataFrame([
    {
        "Symbol": r["ticker"],
        "Name": get_company_name(r["ticker"]),
        "Under Control": r["under_control"],
        "Drift": r["drift_direction"],
        "Buy": r["buy_signal"],
        "Sell": r["sell_signal"],
        "Band Width": r["band_width"],
        "Confidence": r["confidence"],
        "Days Under Control": r["days_under_control"],
        "RSI": r["indicators"]["rsi"],
        "Slope 5D": r["indicators"]["slope_5d"],
        "Price": r["indicators"]["price"],
        "MA20": r["indicators"]["ma20"],
    }
    for r in results
])

df_sorted = df.sort_values(
    by=["Under Control", "Days Under Control", "Band Width", "Confidence"],
    ascending=[False, False, False, False]
).reset_index(drop=True)

# ---------------------------------------
# TOP 20 UNDER CONTROL
# ---------------------------------------
df_top = df_sorted[df_sorted["Under Control"] == True].head(20)

st.subheader("📊 Top 20 Under‑Control Trading Candidates")
st.dataframe(df_top, use_container_width=True)

# ---------------------------------------
# BUY / SELL LOGIC
# ---------------------------------------
st.subheader("🔄 Portfolio Rotation Recommendations")

top_symbols = df_top["Symbol"].tolist()

sell_candidates = [s for s in tracked if s not in top_symbols]
buy_candidates = [s for s in top_symbols if s not in tracked]

# ---------------------------------------
# SELL — FULL DATA TABLE + CHART
# ---------------------------------------
st.write("### ❌ SELL Candidates (Tracked but NOT Under Control)")

df_sell = df[df["Symbol"].isin(sell_candidates)]

if df_sell.empty:
    st.success("No SELL candidates — all tracked stocks are under control.")
else:
    st.dataframe(df_sell, use_container_width=True)

    sell_choice = st.selectbox(
        "Select a SELL candidate to view SPC chart",
        df_sell["Symbol"].tolist(),
        format_func=lambda x: f"{x} — {get_company_name(x)}"
    )
    st.write(f"#### SPC Chart — SELL Candidate: {sell_choice} — {get_company_name(sell_choice)}")
    plot_spc_chart(sell_choice, window=window, control_k=control_k)

# ---------------------------------------
# BUY — ONLY DRIFT = DOWN + CHART
# ---------------------------------------
st.write("### ✅ BUY Candidates (Under Control, NOT Tracked, Drift = 'down')")

df_buy = df_top[
    (df_top["Symbol"].isin(buy_candidates)) &
    (df_top["Buy"] == True)
]

if df_buy.empty:
    st.info("No BUY candidates — none are drifting down.")
else:
    df_buy["Name"] = df_buy["Symbol"].apply(get_company_name)
    st.dataframe(df_buy, use_container_width=True)

    total_cost = df_buy["Price"].sum()
    st.write(f"💰 **Cost to buy 1 share of each BUY candidate:** ${total_cost:,.2f}")

    buy_choice = st.selectbox(
        "Select a BUY candidate to view SPC chart",
        df_buy["Symbol"].tolist(),
        format_func=lambda x: f"{x} — {get_company_name(x)}"
    )
    st.write(f"#### SPC Chart — BUY Candidate: {buy_choice} — {get_company_name(buy_choice)}")
    plot_spc_chart(buy_choice, window=window, control_k=control_k)

# ---------------------------------------
# UPDATE TRACKED LIST AUTOMATICALLY
# ---------------------------------------
if st.button("🔁 Update 'Stocks Being Tracked' Automatically"):
    new_tracked = [s for s in tracked if s not in sell_candidates] + df_buy["Symbol"].tolist()
    new_tracked = sorted(list(set(new_tracked)))
    save_tracked_stocks(new_tracked)
    st.success("Tracked list updated! Refresh the page to see changes.")

# ---------------------------------------
# DETAIL VIEW (FROM TOP 20)
# ---------------------------------------
st.subheader("🔍 Detailed View (Top 20)")

selected_symbol = st.selectbox(
    "Select a symbol for detailed indicators",
    df_top["Symbol"].tolist(),
    format_func=lambda x: f"{x} — {get_company_name(x)}"
)

detail = next(r for r in results if r["ticker"] == selected_symbol)

st.write("### Indicators")
st.json(detail["indicators"])

# ---------------------------------------
# DEBUG PANEL — WHY BUY FAILED OR SUCCEEDED
# ---------------------------------------
st.write("### Debug Panel — BUY Rule Breakdown")

debug_info = {
    "Price in Lower Zone (Bottom 5%)": (
        detail["indicators"]["price"] <= 
        detail["indicators"]["lower_limit"] + 
        0.05 * (detail["indicators"]["upper_limit"] - detail["indicators"]["lower_limit"])
    ),
    "Drift is Down": (detail["drift_direction"] == "down"),
    "Under Control": detail["under_control"],
    "RSI < 60": (detail["indicators"]["rsi"] is not None and detail["indicators"]["rsi"] < 60),
}

st.json(debug_info)

# ---------------------------------------
# KEY DEFINITIONS FOR USER TESTERS
# ---------------------------------------
st.write("### 📘 Key Definitions (How This Model Works)")

st.markdown("""
**Under Control**  
A stock is under control when its price stays between the SPC control limits (UCL and LCL) and volatility remains stable.  
This indicates predictable oscillation behavior.

**Days Under Control**  
The number of consecutive days the stock has remained inside the control limits with stable volatility.  
Higher values indicate stronger stability and more reliable oscillation cycles.

**Drift Direction**  
The short-term slope of price over the last 5 days.  
- **Down** → price is declining  
- **Up** → price is rising  
- **Flat** → no meaningful movement  

**Band Width**  
The distance between UCL and LCL.  
Wider bands indicate larger oscillation amplitude and more profit potential.

**Bottom 5% Zone (BUY Zone)**  
The lowest 5% of the control band.  
A BUY signal requires the price to be inside this zone.

**BUY Signal Requirements**  
A stock triggers a BUY signal only if:  
- Price is in the bottom 5% of the band  
- Drift direction is down  
- Stock is under control  
- RSI < 60  

**SELL Signal Requirements**  
A stock triggers a SELL signal when:  
- Price is in the top 10% of the band  
- Drift direction is up  
- Stock is under control  
- RSI > 40  
""")


st.write("### Signals")
st.write(f"**Under Control:** {detail['under_control']}")
st.write(f"**Drift:** {detail['drift_direction']}")
st.write(f"**Buy Signal:** {detail['buy_signal']}")
st.write(f"**Sell Signal:** {detail['sell_signal']}")
st.write(f"**Confidence:** {detail['confidence']:.2f}")
st.write(f"**Band Width:** {detail['band_width']:.2f}")
st.write(f"**Days Under Control:** {detail['days_under_control']}")