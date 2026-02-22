import streamlit as st
import pandas as pd

from stock_engine import (
    analyze_stock,
    get_fundamentals,
    compute_fundamental_quality,
    compute_composite_score,
    normalize_ticker,
)




# -----------------------------
# PAGE 3 — OPPORTUNITY ANALYSIS
# -----------------------------

st.title("🚀 Opportunity Analysis")

st.write("""
This page applies your **timing engine** (Momentum, Risk, Signal, Traffic Light) 
to the companies you selected in **Page 2 — Financials Explorer**.

This is where fundamentals meet timing.
""")


# --- Step 1: Check if user selected companies in Page 2 ---
if "selected_financials" not in st.session_state or not st.session_state["selected_financials"]:
    st.warning("No companies selected. Please go to **Page 2 — Financials Explorer** and choose companies first.")
    st.stop()

selected = st.session_state["selected_financials"]

st.write(f"Companies selected for analysis: **{len(selected)}**")
st.write(", ".join(selected))


# --- Step 2: Run timing engine on selected companies ---
if st.button("🔍 Run Opportunity Analysis"):

    progress = st.progress(0)
    results = []

    for i, sym in enumerate(selected):
        progress.progress((i + 1) / len(selected))

        sym_norm = normalize_ticker(sym)

        try:
            # Fetch fundamentals (already filtered in Page 2)
            f = get_fundamentals(sym_norm)

            # Run your engine
            data = analyze_stock(sym_norm)
            if not data:
                continue

            signal = data.get("Signal")
            risk = data.get("Risk Score")
            momentum = data.get("Momentum Score")

            # Compute fundamental quality + composite score
            fq = compute_fundamental_quality(f)
            composite = compute_composite_score(momentum, risk, fq)

            row = {
                "Symbol": data.get("Symbol", sym_norm),
                "Company Name": data.get("Company Name"),
                "Signal": signal,
                "Traffic Light": data.get("Traffic Light"),
                "Risk Score": risk,
                "Momentum Score": momentum,
                "Fundamental Quality": fq,
                "Composite Score": composite,
                "Stock Price": data.get("Stock Price"),
                "Avg 3M": data.get("Avg 3M"),
                "Avg 6M": data.get("Avg 6M"),
                "Avg 12M": data.get("Avg 12M"),
                "Avg 24M": data.get("Avg 24M"),
            }

            results.append(row)

        except Exception:
            continue


    # --- Step 3: Display results ---
    if not results:
        st.info("No valid analysis results returned.")
        st.stop()

    df = pd.DataFrame(results)
    df = df.sort_values(by="Composite Score", ascending=False)

    st.subheader("📊 Ranked Opportunities")

    st.dataframe(
        df[
            [
                "Symbol",
                "Company Name",
                "Composite Score",
                "Momentum Score",
                "Risk Score",
                "Fundamental Quality",
                "Signal",
                "Traffic Light",
                "Stock Price",
            ]
        ]
    )

    # --- Optional: Show detailed metrics ---
    st.markdown("---")
    st.subheader("📘 Detailed Metrics")

    st.dataframe(
        df[
            [
                "Symbol",
                "Avg 3M",
                "Avg 6M",
                "Avg 12M",
                "Avg 24M",
                "Risk Score",
                "Momentum Score",
                "Fundamental Quality",
            ]
        ]
    )


# --- Explanation section ---
st.markdown("""
---
### 📘 How This Page Works

This page applies your **timing logic** to companies selected in Page 2:

#### **1. Your Engine**
- Momentum Score  
- Risk Score  
- BUY / HOLD / AVOID Signal  
- Traffic Light  

#### **2. Composite Score**
Weighted ranking:
- 50% Momentum  
- 30% Fundamental Quality  
- 20% (100 − Risk)  

#### **3. Ranking**
Companies are sorted by Composite Score to highlight the strongest opportunities.
""")