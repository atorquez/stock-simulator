import streamlit as st
import pandas as pd
from stock_engine import analyze_stock

# Page layout
st.set_page_config(layout="wide")

# Financial-style theme
st.markdown("""
    <style>
        .stApp {
            background-color: #0d1117;
            color: #e6e6e6;
        }
        label, .stTextInput label {
            font-weight: 600 !important;
            color: #e6e6e6 !important;
        }
        thead tr th {
            font-size: 18px !important;
            font-weight: 800 !important;
            background-color: #161b22 !important;
            color: #f1c40f !important;
            border-bottom: 2px solid #f1c40f !important;
        }
        tbody tr td {
            font-size: 16px !important;
            color: #e6e6e6 !important;
        }
        tbody tr:hover {
            background-color: #1f2937 !important;
        }
        .stButton>button {
            background-color: #f1c40f;
            color: black;
            font-weight: 700;
            border-radius: 6px;
            height: 3em;
            width: 10em;
        }
        .stButton>button:hover {
            background-color: #d4ac0d;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📈 Stock Analyzer App")

st.write("Stock tickers (comma-separated):")

# Input
user_input = st.text_input(
    "Stock Symbols",
    "WCC, TSLA, MSFT, QCOM, SPCE, DIS, PDD, MOMO, TME, WB, NEOG, BZUN, OBE, NVDA"
)

# Main logic
if st.button("Analyze"):
    symbols = [s.strip().upper() for s in user_input.split(",") if s.strip()]

    results = []
    for sym in symbols:
        result = analyze_stock(sym)
        if result:
            results.append(result)
        else:
            st.warning(f"Could not fetch data for {sym}")

    if len(results) > 0:
        df = pd.DataFrame(results)

        # Ensure Company Name column exists
        if "Company Name" not in df.columns:
            df["Company Name"] = "N/A"

        # Clean symbol for summary table
        df["Symbol_Clean"] = df["Symbol"] + " — " + df["Company Name"]

        # HTML symbol with hover tooltip for main table
        df["Symbol_HTML"] = df.apply(
            lambda row: f'<span title="{row["Company Name"]}">{row["Symbol"]}</span>',
            axis=1
        )

        # Build main display table
        df_display = df.copy()
        df_display["Symbol"] = df_display["Symbol_HTML"]
        df_display = df_display.drop(columns=["Company Name", "Symbol_HTML", "Symbol_Clean"])

        # Rename for readability (APPLY TO BOTH TABLES)
        rename_map = {
            "Stock Price": "Price",
            "Market Cap (M)": "Mkt Cap",
            "Dividend Yield": "Div Yld",
            "PE Ratio": "PE",
            "Momentum Score": "Momentum",
            "Risk Score": "Risk",
        }

        df = df.rename(columns=rename_map)
        df_display = df_display.rename(columns=rename_map)

        # MAIN TABLE (hover-enabled)
        st.subheader("📊 Analysis Results")
        st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

        # SUMMARY TABLE (clean text)
        st.subheader("🔎 Summary View")

        summary_cols = ["Symbol_Clean", "Signal", "Traffic Light", "Risk", "Momentum", "Price", "Mkt Cap"]
        df_summary = df[summary_cols].rename(columns={"Symbol_Clean": "Symbol"})

        st.table(df_summary)

        # KEY METRICS REFERENCE
        st.markdown("""
---  
### 📘 Key Metrics Reference

**🟢🟡🔴 Traffic Light**  
Quick visual summary of the stock’s overall condition:  
- 🟢 **Green** — Strong fundamentals and positive trend  
- 🟡 **Yellow** — Mixed signals; proceed with caution  
- 🔴 **Red** — Elevated risk or long‑term decline  

**⚠️ Risk Score**  
Measures volatility, drawdowns, and price stability:  
- 0–20 → Low risk (stable, predictable)  
- 20–40 → Moderate risk (normal market behavior)  
- 40–60 → High risk (large swings, sensitive to news)  
- 60+ → Very high / speculative (unstable or declining)  

**⚡ Momentum Score**  
Captures recent price strength vs. historical averages:  
- 60–100 → Strong upward trend  
- 40–60 → Neutral or stabilizing  
- 0–40 → Weak or declining trend  

**📌 Signal**  
Synthesized view of fundamentals, momentum, and risk:  
- **BUY** — Strong fundamentals + positive trend  
- **HOLD** — Neutral outlook  
- **SPECULATIVE** — High risk; aggressive investors only  
- **AVOID** — Weak fundamentals or long‑term decline  
""")