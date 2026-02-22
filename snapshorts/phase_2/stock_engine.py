import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


SHORT_WINDOW = 20
LONG_WINDOW = 50

# ---------------------------------------
# TICKER NORMALIZATION (fix Yahoo issues)
# ---------------------------------------
TICKER_NORMALIZATION = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

def normalize_ticker(ticker: str) -> str:
    return TICKER_NORMALIZATION.get(ticker, ticker)


# ---------------------------------------
# FUNDAMENTAL EXTRACTION
# ---------------------------------------
def get_fundamentals(ticker: str):
    """Fetch key fundamentals safely."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception:
        return {}

    try:
        return {
            "Market Cap": info.get("marketCap"),
            "PE Ratio": info.get("trailingPE"),
            "Forward PE": info.get("forwardPE"),
            "PEG Ratio": info.get("pegRatio"),
            "ROE": info.get("returnOnEquity"),
            "Profit Margin": info.get("profitMargins"),
            "Operating Margin": info.get("operatingMargins"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Earnings Growth": info.get("earningsGrowth"),
            "Debt to Equity": info.get("debtToEquity"),
            "Free Cash Flow": info.get("freeCashflow"),
            "Dividend Yield": info.get("dividendYield"),
        }
    except Exception:
        return {}


# ---------------------------------------
# FUNDAMENTAL FILTER (broad quality gate)
# ---------------------------------------
def passes_fundamentals(f):
    """Broad quality filter — removes weak companies only."""
    mkt_cap = f.get("Market Cap") or 0
    profit_margin = f.get("Profit Margin") or 0
    roe = f.get("ROE") or 0
    rev_growth = f.get("Revenue Growth")
    debt_to_equity = f.get("Debt to Equity")
    fcf = f.get("Free Cash Flow")

    if mkt_cap < 5_000_000_000:
        return False
    if profit_margin <= 0:
        return False
    if roe <= 0:
        return False
    if rev_growth is None or rev_growth <= 0:
        return False
    if debt_to_equity is None or debt_to_equity >= 200:
        return False
    if fcf is None or fcf <= 0:
        return False

    return True


# ---------------------------------------
# FUNDAMENTAL QUALITY SCORE (0–100)
# ---------------------------------------
def compute_fundamental_quality(f):
    score = 0

    # ROE
    roe = f.get("ROE") or 0
    if roe > 0:
        score += min(roe * 100, 30)

    # Profit Margin
    pm = f.get("Profit Margin") or 0
    if pm > 0:
        score += min(pm * 100, 30)

    # Revenue Growth
    rg = f.get("Revenue Growth") or 0
    if rg > 0:
        score += min(rg * 100, 20)

    # Debt to Equity
    de = f.get("Debt to Equity")
    if de is not None and de > 0:
        if de < 50:
            score += 20
        elif de < 100:
            score += 10
        elif de < 200:
            score += 5

    return min(score, 100)


# ---------------------------------------
# COMPOSITE SCORE (Momentum + Fundamentals + Risk)
# ---------------------------------------
def compute_composite_score(momentum_score, risk_score, fundamental_quality):
    m = momentum_score or 0
    r = risk_score or 100
    fq = fundamental_quality or 0

    composite = (
        0.5 * m +
        0.3 * fq +
        0.2 * (100 - r)
    )
    return composite


# ---------------------------------------
# PRICE HISTORY
# ---------------------------------------
def fetch_history(symbol, period="24mo"):
    """Fetch historical data safely."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data is None or data.empty:
            return None
        return data
    except Exception:
        return None


# ---------------------------------------
# MOVING AVERAGES
# ---------------------------------------
def compute_multi_averages(data):
    close = data["Close"]

    def avg(days):
        return close.iloc[-days:].mean() if len(close) >= days else close.mean()

    return {
        "avg_3m": round(avg(63), 2),
        "avg_6m": round(avg(126), 2),
        "avg_12m": round(avg(252), 2),
        "avg_24m": round(avg(504), 2),
    }


def is_structural_decline(avgs):
    return (
        avgs["avg_3m"] < avgs["avg_6m"] <
        avgs["avg_12m"] < avgs["avg_24m"]
    )


# ---------------------------------------
# RISK SCORE
# ---------------------------------------
def compute_risk_score(data):
    close = data["Close"]
    returns = close.pct_change().dropna()
    if returns.empty:
        return 50

    vol = returns.std()
    score = min(100, max(0, vol * 1000))
    return round(score, 1)


# ---------------------------------------
# MOMENTUM SCORE
# ---------------------------------------
def compute_momentum_score(data, avgs):
    close = data["Close"]
    end_price = close.iloc[-1]

    if avgs["avg_12m"] == 0:
        return 50

    rel_to_12m = (end_price - avgs["avg_12m"]) / avgs["avg_12m"]
    rel_3m_vs_12m = (avgs["avg_3m"] - avgs["avg_12m"]) / avgs["avg_12m"]

    raw = 50 + 100 * (0.5 * rel_to_12m + 0.5 * rel_3m_vs_12m)
    score = min(100, max(0, raw))
    return round(score, 1)


# ---------------------------------------
# SIGNAL + TRAFFIC LIGHT
# ---------------------------------------
def traffic_light(signal, risk_score, momentum_score):
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

    if is_structural_decline(avgs):
        return "AVOID - STRUCTURAL DECLINE", avgs

    if end_price < 0.5 * start_price:
        return "AVOID - LONG TERM DECLINE", avgs

    if end_price < 5:
        return "SPECULATIVE - HIGH RISK", avgs

    price_change_pct = (end_price - avgs["avg_12m"]) / avgs["avg_12m"]

    if price_change_pct <= -0.2:
        return "BUY", avgs
    else:
        return "HOLD", avgs


# ---------------------------------------
# INDICATORS
# ---------------------------------------
def fetch_indicators(symbol):
    """Fetch price + fundamentals safely."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
    except Exception:
        return {}

    try:
        hist = ticker.history(period="1d")
        price = hist["Close"].iloc[-1] if not hist.empty else None
    except Exception:
        price = None

    try:
        return {
            "Stock Price": round(float(price), 2) if price else None,
            "Market Cap (M)": f"{info.get('marketCap', 0) / 1_000_000:,.0f}" if info.get("marketCap") else None,
            "PE Ratio": round(info.get("trailingPE"), 2) if info.get("trailingPE") else None,
            "EPS": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {}


# ---------------------------------------
# FULL ANALYSIS PIPELINE
# ---------------------------------------
def analyze_stock(symbol):
    """Full pipeline for one stock."""
    symbol = normalize_ticker(symbol)

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