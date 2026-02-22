<<<<<<< HEAD
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


def compute_days_under_control(close, ma, upper, lower, std, window):
    """
    Count how many consecutive days (including today)
    the stock has stayed inside control limits AND
    maintained stable volatility.
    """
    days = 0
    n = len(close)

    for i in range(n - 1, -1, -1):
        # Stop if limits or std are missing
        if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]) or pd.isna(std.iloc[i]):
            break

        price_i = close.iloc[i]
        u_i = upper.iloc[i]
        l_i = lower.iloc[i]

        # Volatility stability check
        recent_std_i = std.iloc[max(0, i - window + 1): i + 1].mean()
        if std.iloc[i] > recent_std_i * 1.5:
            break

        # Must be inside the band
        if not (l_i <= price_i <= u_i):
            break

        days += 1

    return days

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

# ---------------------------------------
# SHORT-TERM TRADING ENGINE (SPC-STYLE)
# ---------------------------------------

def compute_RSI(series, period=14):
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_control_limits(close, window=20, k=2.0):
    """Compute SPC-style control limits (centerline ± k * std)."""
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()

    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower, std


def compute_drift(close, lookback=5):
    """Compute simple drift direction based on lookback slope."""
    if len(close) <= lookback:
        return 0.0, "flat"

    slope = close.iloc[-1] - close.iloc[-(lookback + 1)]
    if slope > 0:
        direction = "up"
    elif slope < 0:
        direction = "down"
    else:
        direction = "flat"
    return slope, direction


def analyze_trading_signals(
    ticker: str,
    window: int = 20,
    control_k: float = 2.0,
    max_hold_days: int = 10
) -> dict | None:
    """
    Short-term trading analysis using SPC-style control logic.

    Returns a dict with:
        - under_control
        - drift_direction
        - buy_signal
        - sell_signal
        - confidence
        - band_width
        - indicators (price, ma20, limits, rsi, slope, bollinger_position)
    """
    symbol = normalize_ticker(ticker)

    # Use existing history fetcher for consistency
    data = fetch_history(symbol, period="6mo")
    if data is None or data.empty:
        return None

    close = data["Close"].dropna()
    if close.empty:
        return None

    # --- Control limits (SPC-style) ---
    ma, upper_limit, lower_limit, std = compute_control_limits(
        close, window=window, k=control_k
    )

    days_under_control = compute_days_under_control(
    close, ma, upper_limit, lower_limit, std, window
    )
    
    # Use last valid point
    ma20 = ma.iloc[-1]
    ucl = upper_limit.iloc[-1]
    lcl = lower_limit.iloc[-1]
    price = close.iloc[-1]

    # Band width
    band_width = float(ucl - lcl) if pd.notna(ucl) and pd.notna(lcl) else 0.0

    # --- Under-control condition ---
    recent_std = std.iloc[-window:].mean() if std.iloc[-window:].notna().any() else std.mean()
    under_control = (
        pd.notna(ucl)
        and pd.notna(lcl)
        and (lcl <= price <= ucl)
        and pd.notna(std.iloc[-1])
        and std.iloc[-1] <= recent_std * 1.5
    )

    # --- Drift direction (5-day slope) ---
    slope_5d, drift = compute_drift(close, lookback=5)

    # --- RSI ---
    rsi_series = compute_RSI(close, period=14)
    rsi = float(rsi_series.iloc[-1]) if not rsi_series.dropna().empty else None

    # --- Bollinger-style position inside band (0–1) ---
    if pd.notna(ucl) and pd.notna(lcl) and (ucl - lcl) > 0:
        bollinger_pos = float((price - lcl) / (ucl - lcl))
    else:
        bollinger_pos = 0.5

    # ============================================================
    #  SPC-STYLE BUY RULE (Bottom-Zone Entry)
    # ============================================================
    if band_width > 0:
        lower_zone_threshold = lcl + 0.05 * (ucl - lcl)
        price_in_lower_zone = price <= lower_zone_threshold
    else:
        price_in_lower_zone = False

    drift_is_down = (drift == "down")
    is_under_control = (under_control is True)
    rsi_ok_for_buy = (rsi is not None and rsi < 60)

    buy_signal = (
        price_in_lower_zone and
        drift_is_down and
        is_under_control and
        rsi_ok_for_buy
    )

    # ============================================================
    #  SPC-STYLE SELL RULE (Top-Zone Exit)
    # ============================================================
    if band_width > 0:
        upper_zone_threshold = ucl - 0.10 * (ucl - lcl)
        price_in_upper_zone = price >= upper_zone_threshold
    else:
        price_in_upper_zone = False

    drift_is_up = (drift == "up")
    rsi_ok_for_sell = (rsi is not None and rsi > 40)

    sell_signal = (
        price_in_upper_zone and
        drift_is_up and
        is_under_control and
        rsi_ok_for_sell
    )

    # --- Max hold window (algorithmic lookback using bottom-zone BUY rule) ---
    days_since_buy = None
    if len(close) > window + max_hold_days and band_width > 0:
        for i in range(1, max_hold_days + 1):
            sub_close = close.iloc[: -i + 1] if i > 1 else close
            sub_ma, sub_u, sub_l, sub_std = compute_control_limits(
                sub_close, window=window, k=control_k
            )

            p_i = sub_close.iloc[-1]
            ma_i = sub_ma.iloc[-1]
            u_i = sub_u.iloc[-1]
            l_i = sub_l.iloc[-1]

            if pd.isna(u_i) or pd.isna(l_i) or (u_i - l_i) <= 0:
                continue

            recent_std_i = sub_std.iloc[-window:].mean() if sub_std.iloc[-window:].notna().any() else sub_std.mean()
            under_control_i = (
                pd.notna(u_i)
                and pd.notna(l_i)
                and (l_i <= p_i <= u_i)
                and pd.notna(sub_std.iloc[-1])
                and sub_std.iloc[-1] <= recent_std_i * 1.5
            )

            slope_i, drift_i = compute_drift(sub_close, lookback=5)

            rsi_i_series = compute_RSI(sub_close, period=14)
            rsi_i = float(rsi_i_series.iloc[-1]) if not rsi_i_series.dropna().empty else None

            lower_zone_threshold_i = l_i + 0.10 * (u_i - l_i)
            price_in_lower_zone_i = p_i <= lower_zone_threshold_i
            drift_is_down_i = (drift_i == "down")
            rsi_ok_for_buy_i = (rsi_i is not None and rsi_i < 60)

            buy_i = (
                price_in_lower_zone_i and
                drift_is_down_i and
                under_control_i and
                rsi_ok_for_buy_i
            )

            if buy_i:
                days_since_buy = i - 1
                break

    if days_since_buy is not None and days_since_buy > max_hold_days:
        sell_signal = True

    # --- Confidence score (aligned with bottom-zone geometry) ---
    confidence_components = [
        1.0 if under_control else 0.0,
        1.0 if drift == "down" else 0.0,
        1.0 if price_in_lower_zone else 0.0,
        1.0 if (rsi is not None and 40 <= rsi <= 60) else 0.0,
    ]
    confidence = float(sum(confidence_components) / len(confidence_components))

    return {
        "ticker": symbol,
        "under_control": under_control,
        "drift_direction": drift,
        "buy_signal": buy_signal,
        "sell_signal": sell_signal,
        "days_since_buy": days_since_buy,
        "days_under_control": days_under_control,
        "confidence": confidence,
        "band_width": band_width,
        "indicators": {
            "price": float(price),
            "ma20": float(ma20),
            "upper_limit": float(ucl),
            "lower_limit": float(lcl),
            "rsi": float(rsi) if rsi is not None else None,
            "slope_5d": float(slope_5d),
            "bollinger_position": bollinger_pos,
        },
=======
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
>>>>>>> d338b99beb2aa3de7fca642818511d14821c59a4
    }