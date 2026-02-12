import yfinance as yf
import pandas as pd
import numpy as np

SHORT_WINDOW = 20
LONG_WINDOW = 50


def fetch_history(symbol, period="24mo"):
    """Fetch historical data safely."""
    try:
        data = yf.Ticker(symbol).history(period=period)
        return data if not data.empty else None
    except:
        return None


def compute_multi_averages(data):
    close = data["Close"]

    def avg(days):
        return close.iloc[-days:].mean() if len(close) >= days else close.mean()

    return {
        "avg_3m": round(avg(63), 2),     # 3 months
        "avg_6m": round(avg(126), 2),    # 6 months
        "avg_12m": round(avg(252), 2),   # 12 months
        "avg_24m": round(avg(504), 2),   # 24 months
    }


def is_structural_decline(avgs):
    return (
        avgs["avg_3m"] < avgs["avg_6m"] <
        avgs["avg_12m"] < avgs["avg_24m"]
    )


def compute_risk_score(data):
    """Simple volatility-based risk score (0–100)."""
    close = data["Close"]
    returns = close.pct_change().dropna()
    if returns.empty:
        return 50

    vol = returns.std()  # daily volatility
    # Map vol roughly into 0–100
    score = min(100, max(0, vol * 1000))
    return round(score, 1)


def compute_momentum_score(data, avgs):
    """Momentum based on price vs 12m avg and 3m vs 12m."""
    close = data["Close"]
    end_price = close.iloc[-1]

    if avgs["avg_12m"] == 0:
        return 50

    rel_to_12m = (end_price - avgs["avg_12m"]) / avgs["avg_12m"]
    rel_3m_vs_12m = (avgs["avg_3m"] - avgs["avg_12m"]) / avgs["avg_12m"]

    raw = 50 + 100 * (0.5 * rel_to_12m + 0.5 * rel_3m_vs_12m)
    score = min(100, max(0, raw))
    return round(score, 1)


def traffic_light(signal, risk_score, momentum_score):
    """
    Return a simple traffic-light label:
    - Green: BUY or strong momentum, moderate risk
    - Yellow: HOLD or mixed
    - Red: AVOID / SPECULATIVE / very high risk
    """
    sig_upper = signal.upper()

    if "AVOID" in sig_upper:
        return "🔴 Red"
    if "SPECULATIVE" in sig_upper:
        return "🔴 Red"

    if "BUY" in sig_upper and momentum_score >= 60 and risk_score <= 60:
        return "🟢 Green"

    if risk_score >= 80:
        return "🔴 Red"

    return "🟡 Yellow"


def generate_signal(data):
    close = data["Close"]
    start_price = close.iloc[0]
    end_price = close.iloc[-1]

    avgs = compute_multi_averages(data)

    # 1. Structural decline override
    if is_structural_decline(avgs):
        base_signal = "AVOID - STRUCTURAL DECLINE"
        return base_signal, avgs

    # 2. Long-term collapse override
    if end_price < 0.5 * start_price:
        base_signal = "AVOID - LONG TERM DECLINE"
        return base_signal, avgs

    # 3. Penny stock risk filter
    if end_price < 5:
        base_signal = "SPECULATIVE - HIGH RISK"
        return base_signal, avgs

    # 4. Standard BUY/HOLD logic
    price_change_pct = (end_price - avgs["avg_12m"]) / avgs["avg_12m"]

    if price_change_pct <= -0.2:
        base_signal = "BUY"
    else:
        base_signal = "HOLD"

    return base_signal, avgs


def fetch_indicators(symbol):
    """Fetch price + fundamentals + company name."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        hist = ticker.history(period="1d")
        price = hist["Close"].iloc[-1] if not hist.empty else None

        return {
            "Company Name": info.get("longName") or info.get("shortName") or "N/A",
            "Stock Price": round(float(price), 2) if price else None,
            "Market Cap (M)": f"{info.get('marketCap', 0) / 1_000_000:,.0f}" if info.get("marketCap") else None,
            "PE Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            "EPS": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
        }
    except:
        return {"Company Name": "N/A"}


def analyze_stock(symbol):
    """Full pipeline for one stock."""
    data = fetch_history(symbol)
    if data is None:
        return None

    base_signal, avgs = generate_signal(data)
    risk_score = compute_risk_score(data)
    momentum_score = compute_momentum_score(data, avgs)
    light = traffic_light(base_signal, risk_score, momentum_score)
    indicators = fetch_indicators(symbol)

    return {
        "Symbol": symbol,
        "Signal": base_signal,
        "Traffic Light": light,
        "Risk Score": risk_score,
        "Momentum Score": momentum_score,
        "Avg 3M": avgs["avg_3m"],
        "Avg 6M": avgs["avg_6m"],
        "Avg 12M": avgs["avg_12m"],
        "Avg 24M": avgs["avg_24m"],
        **indicators
    }