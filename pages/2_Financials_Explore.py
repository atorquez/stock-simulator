import streamlit as st
import pandas as pd
import yfinance as yf
import re

from stock_engine import (
    get_fundamentals,
    passes_fundamentals,
    normalize_ticker,
)

from data.sp500_list import sp500
from data.nasdaq100_list import nasdaq100



# -----------------------------------------
# Helper: Fetch company name safely
# -----------------------------------------
def get_company_name(ticker: str):
    """Fetch company name safely."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ""
    except:
        return ""


# -----------------------------
# PAGE 2 — FINANCIALs EXPLORER
# -----------------------------
st.title("📊 Financials Explorer")

st.write("""
Use this page to explore companies based on **fundamental quality**.
This is your discovery layer — find strong companies first, then analyze timing in Page 3.
""")


# --- Step 1: Choose universe ---
universe_option = st.selectbox(
    "Select a stock universe:",
    ["S&P 500", "NASDAQ 100", "Custom List"]
)

def load_universe(option):
    if option == "S&P 500":
        return sp500
    elif option == "NASDAQ 100":
        return nasdaq100
    else:
        return []

tickers = load_universe(universe_option)

if universe_option == "Custom List":
    user_input = st.text_input(
        "Enter tickers (comma-separated):",
        "AAPL, MSFT, NVDA, TSLA"
    )
    tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

st.write(f"Selected universe size: **{len(tickers)}** tickers")


# --- Step 2 + Step 3: Fetch fundamentals + filter (inside button) ---
if st.button("📥 Load Fundamentals"):

    progress = st.progress(0)
    rows = []

    for i, sym in enumerate(tickers):
        progress.progress((i + 1) / len(tickers))

        sym_norm = normalize_ticker(sym)

        try:
            f = get_fundamentals(sym_norm)
            if not f:
                continue

            company_name = get_company_name(sym_norm)

            row = {
                "Symbol": sym_norm,
                "Company Name": company_name,
                "Market Cap": f.get("Market Cap"),
                "PE Ratio": f.get("PE Ratio"),
                "ROE": f.get("ROE"),
                "Profit Margin": f.get("Profit Margin"),
                "Revenue Growth": f.get("Revenue Growth"),
                "Debt to Equity": f.get("Debt to Equity"),
                "Free Cash Flow": f.get("Free Cash Flow"),
                "Passes Fundamentals": passes_fundamentals(f),
            }

            rows.append(row)

        except Exception:
            continue

    if not rows:
        st.warning("No fundamentals could be loaded.")
        st.stop()

    df = pd.DataFrame(rows)

    st.subheader("📘 Fundamental Metrics")
    st.dataframe(df)

    # --- Step 3: Filter options ---
    st.markdown("---")
    st.subheader("🔎 Filter Companies")

    min_mktcap = st.number_input("Minimum Market Cap ($B)", 1, 500, 5)
    min_roe = st.number_input("Minimum ROE (%)", 0, 100, 5)
    min_profit = st.number_input("Minimum Profit Margin (%)", 0, 100, 5)
    min_rev_growth = st.number_input("Minimum Revenue Growth (%)", -50, 100, 0)
    max_debt = st.number_input("Maximum Debt-to-Equity", 0, 1000, 200)

    filtered = df[
        (df["Market Cap"] >= min_mktcap * 1_000_000_000) &
        (df["ROE"] >= min_roe / 100) &
        (df["Profit Margin"] >= min_profit / 100) &
        (df["Revenue Growth"] >= min_rev_growth / 100) &
        (df["Debt to Equity"] <= max_debt)
    ]

    st.write(f"Companies passing filters: **{len(filtered)}**")
    st.dataframe(filtered)

    # Save filtered list for Step 4
    st.session_state["filtered_df"] = filtered


# --- Step 4: Select companies for Page 3 (OUTSIDE button) ---
if "filtered_df" in st.session_state:

    st.markdown("---")
    st.subheader("📌 Select Companies for Opportunity Analysis")

    filtered = st.session_state["filtered_df"]

    selection_labels = filtered.apply(
        lambda row: f"{row['Symbol']} — {row['Company Name']}",
        axis=1
    ).tolist()

    selected = st.multiselect(
        "Choose companies to analyze in Page 3:",
        selection_labels
    )

    if selected:
        selected_symbols = [
            re.split(r"\s+[—–-]\s+", s)[0].strip()
            for s in selected
        ]

        st.session_state["selected_financials"] = selected_symbols
        st.success(f"Saved: {', '.join(selected_symbols)}")