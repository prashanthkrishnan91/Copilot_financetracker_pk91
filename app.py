# streamlit_recommendation_engine.py
import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import yfinance as yf
import os
from datetime import datetime

# -------------------------
# Configuration and Helpers
# -------------------------
HISTORY_PATH = "history.csv"
BIWEEKLY_DEPOSIT = 900.0

st.set_page_config(page_title="Portfolio Recommendation Engine", layout="wide")

# -------------------------
# Utility Functions
# -------------------------
def parse_pdf(pdf_file):
    """
    Parse Robinhood PDF monthly statement to extract holdings and cash balance.
    Returns two DataFrames: holdings_df with columns ['ticker','shares','price'] and
    cash_balance float.
    Uses pdfplumber to handle table layouts.
    """
    holdings = []
    cash_balance = 0.0
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                # Try to extract tables
                tables = page.extract_tables()
                for table in tables:
                    # Normalize table rows
                    df = pd.DataFrame(table[1:], columns=table[0])
                    cols = [c.lower() for c in df.columns]
                    # Heuristics for holdings table
                    if any("symbol" in c or "ticker" in c for c in cols) and any("quantity" in c or "shares" in c for c in cols):
                        # Attempt to standardize
                        col_map = {}
                        for c in df.columns:
                            lc = c.lower()
                            if "symbol" in lc or "ticker" in lc:
                                col_map[c] = "ticker"
                            if "quantity" in lc or "shares" in lc:
                                col_map[c] = "shares"
                            if "price" in lc or "last" in lc or "market price" in lc:
                                col_map[c] = "price"
                        df = df.rename(columns=col_map)
                        if "ticker" in df.columns and "shares" in df.columns:
                            # Clean rows
                            df = df[["ticker","shares"] + ([c for c in ["price"] if c in df.columns])]
                            df["ticker"] = df["ticker"].str.strip().str.upper()
                            df["shares"] = df["shares"].replace({',':''}, regex=True).astype(float)
                            if "price" in df.columns:
                                df["price"] = df["price"].replace({'\$':'',',':''}, regex=True).astype(float)
                            else:
                                df["price"] = np.nan
                            holdings.append(df)
                    # Heuristics for cash balance
                    if any("cash" in c.lower() or "available cash" in c.lower() for c in df.columns):
                        # Try to find numeric cash values in the table
                        for col in df.columns:
                            if "cash" in col.lower() or "available" in col.lower():
                                # find numeric-looking entries
                                vals = df[col].replace({'\$':'',',':''}, regex=True)
                                for v in vals:
                                    try:
                                        cash_balance = float(v)
                                        break
                                    except:
                                        continue
    except Exception as e:
        st.warning(f"PDF parsing error: {e}")
    if holdings:
        holdings_df = pd.concat(holdings, ignore_index=True)
        holdings_df = holdings_df.groupby("ticker", as_index=False).agg({"shares":"sum","price":"mean"})
    else:
        holdings_df = pd.DataFrame(columns=["ticker","shares","price"])
    return holdings_df, float(cash_balance)

def fetch_prices(tickers):
    """
    Fetch current market prices using yfinance. Returns dict ticker->price.
    """
    prices = {}
    if len(tickers) == 0:
        return prices
    try:
        data = yf.download(tickers=list(tickers), period="1d", threads=False, progress=False)
        # yfinance returns different shapes for single vs multiple tickers
        if isinstance(data, pd.DataFrame) and "Close" in data.columns:
            # single ticker
            last = data["Close"].iloc[-1]
            prices[list(tickers)[0]] = float(last)
        else:
            # multi ticker
            close = data["Close"].iloc[-1]
            for t in close.index:
                prices[t] = float(close.loc[t])
    except Exception as e:
        st.info("Live price fetch failed, please enter manual prices where needed.")
    return prices

def calculate_drift(holdings_df, targets_df, manual_prices=None):
    """
    holdings_df: columns ['ticker','shares','price' optional]
    targets_df: columns ['ticker','target_pct'] where target_pct is 0-100
    manual_prices: dict ticker->price to override
    Returns DataFrame with current_value, current_pct, target_pct, drift_pct
    """
    df = holdings_df.copy()
    df = df.merge(targets_df, on="ticker", how="outer")
    df["shares"] = df["shares"].fillna(0.0)
    # Determine prices
    tickers = df["ticker"].dropna().unique().tolist()
    live_prices = fetch_prices([t for t in tickers if pd.notna(t)])
    prices = {}
    for t in tickers:
        p = None
        if manual_prices and t in manual_prices and not pd.isna(manual_prices[t]):
            p = float(manual_prices[t])
        elif t in live_prices:
            p = float(live_prices[t])
        else:
            # fallback to price column if present
            row = df[df["ticker"] == t]
            if "price" in row.columns and not row["price"].isna().all():
                p = float(row["price"].dropna().iloc[0])
        prices[t] = p if p is not None else 0.0
    df["price"] = df["ticker"].map(prices)
    df["current_value"] = df["shares"] * df["price"]
    total_value = df["current_value"].sum()
    # If total_value is zero, avoid division by zero
    if total_value == 0:
        df["current_pct"] = 0.0
    else:
        df["current_pct"] = df["current_value"] / total_value * 100.0
    df["target_pct"] = df["target_pct"].fillna(0.0)
    df["drift_pct"] = df["current_pct"] - df["target_pct"]
    df = df.sort_values("drift_pct", ascending=False).reset_index(drop=True)
    return df, total_value

def update_history(entry):
    """
    Append a transaction entry dict to history.csv and update session state.
    entry: dict with keys ['date','ticker','action','amount','notes']
    """
    df_entry = pd.DataFrame([entry])
    if os.path.exists(HISTORY_PATH):
        hist = pd.read_csv(HISTORY_PATH)
        hist = pd.concat([hist, df_entry], ignore_index=True)
    else:
        hist = df_entry
    hist.to_csv(HISTORY_PATH, index=False)
    st.session_state.history = hist

# -------------------------
# Initialize Session State
# -------------------------
if "deposit_enabled" not in st.session_state:
    st.session_state.deposit_enabled = True
if "cash_pool" not in st.session_state:
    st.session_state.cash_pool = 0.0
if "holdings" not in st.session_state:
    st.session_state.holdings = pd.DataFrame(columns=["ticker","shares","price"])
if "targets" not in st.session_state:
    # Example default targets
    st.session_state.targets = pd.DataFrame({
        "ticker": ["AAPL","MSFT","VOO"],
        "target_pct": [30.0, 30.0, 40.0]
    })
if "history" not in st.session_state:
    if os.path.exists(HISTORY_PATH):
        st.session_state.history = pd.read_csv(HISTORY_PATH)
    else:
        st.session_state.history = pd.DataFrame(columns=["date","ticker","action","amount","notes"])

# -------------------------
# Sidebar Manual Overrides
# -------------------------
st.sidebar.header("Data Ingestion")
uploaded_csv = st.sidebar.file_uploader("Upload Robinhood CSV", type=["csv"])
uploaded_pdf = st.sidebar.file_uploader("Upload Robinhood PDF statement", type=["pdf"])
st.sidebar.markdown("**Manual Override**")
manual_ticker = st.sidebar.text_input("Add/Edit Ticker (format: TICKER,shares,price)", value="")
if st.sidebar.button("Apply Manual Ticker"):
    try:
        t, s, p = [x.strip() for x in manual_ticker.split(",")]
        t = t.upper()
        s = float(s)
        p = float(p)
        df = st.session_state.holdings
        if t in df["ticker"].values:
            df.loc[df["ticker"] == t, ["shares","price"]] = [s,p]
        else:
            df = pd.concat([df, pd.DataFrame([{"ticker":t,"shares":s,"price":p}])], ignore_index=True)
        st.session_state.holdings = df
        st.sidebar.success(f"Applied {t}")
    except Exception as e:
        st.sidebar.error("Invalid format. Use: TICKER,shares,price")

# Toggle deposit
st.sidebar.markdown("**Biweekly Deposit**")
st.sidebar.checkbox("Enable $900 Biweekly Deposit", value=st.session_state.deposit_enabled, key="deposit_enabled")

# -------------------------
# Tabs Layout
# -------------------------
tabs = st.tabs(["Dashboard","Action Plan","History"])

# -------------------------
# Dashboard Tab
# -------------------------
with tabs[0]:
    st.header("Dashboard")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Holdings")
        # Ingest CSV
        if uploaded_csv:
            try:
                csv_df = pd.read_csv(uploaded_csv)
                # Expect Robinhood CSV with columns like symbol, quantity, average_buy_price
                if "symbol" in csv_df.columns:
                    csv_df = csv_df.rename(columns={"symbol":"ticker"})
                if "ticker" in csv_df.columns:
                    csv_df["ticker"] = csv_df["ticker"].str.upper()
                    if "quantity" in csv_df.columns:
                        csv_df = csv_df.rename(columns={"quantity":"shares"})
                    if "shares" in csv_df.columns:
                        csv_df = csv_df[["ticker","shares"] + ([c for c in ["price","average_buy_price","last_trade_price"] if c in csv_df.columns])]
                        # Normalize price column
                        if "average_buy_price" in csv_df.columns:
                            csv_df = csv_df.rename(columns={"average_buy_price":"price"})
                        elif "last_trade_price" in csv_df.columns:
                            csv_df = csv_df.rename(columns={"last_trade_price":"price"})
                        csv_df["price"] = pd.to_numeric(csv_df["price"], errors="coerce")
                        csv_df["shares"] = pd.to_numeric(csv_df["shares"], errors="coerce")
                        csv_df = csv_df.groupby("ticker", as_index=False).agg({"shares":"sum","price":"mean"})
                        st.session_state.holdings = csv_df
                        st.success("CSV parsed and loaded")
            except Exception as e:
                st.error(f"CSV parse error: {e}")

        # Ingest PDF
        if uploaded_pdf:
            try:
                holdings_df, cash_balance = parse_pdf(uploaded_pdf)
                if not holdings_df.empty:
                    st.session_state.holdings = holdings_df
                if cash_balance > 0:
                    st.session_state.cash_pool = cash_balance
                st.success("PDF parsed (best-effort). Please verify holdings below.")
            except Exception as e:
                st.error(f"PDF parse error: {e}")

        st.dataframe(st.session_state.holdings)

        st.markdown("**Target Allocation**")
        # Allow editing targets inline
        targets_df = st.session_state.targets.copy()
        targets_df = st.data_editor(targets_df, num_rows="dynamic", use_container_width=True)
        # Normalize targets to sum to 100 if user desires
        if st.button("Normalize Targets to 100%"):
            total = targets_df["target_pct"].sum()
            if total > 0:
                targets_df["target_pct"] = targets_df["target_pct"] / total * 100.0
        st.session_state.targets = targets_df

    with col2:
        st.subheader("Portfolio Metrics")
        # Manual cash input
        manual_cash = st.number_input("Manual Cash Balance", value=float(st.session_state.cash_pool or 0.0), step=100.0)
        st.session_state.cash_pool = float(manual_cash)
        # Add biweekly deposit to displayed available cash if enabled
        deposit_amount = BIWEEKLY_DEPOSIT if st.session_state.deposit_enabled else 0.0
        available_cash = st.session_state.cash_pool + deposit_amount
        # Calculate drift and portfolio value
        manual_prices = {}
        # Provide a small UI to override prices
        if not st.session_state.holdings.empty:
            st.markdown("Manual Price Overrides")
            for t in st.session_state.holdings["ticker"].unique():
                key = f"price_override_{t}"
                val = st.number_input(f"Price for {t}", value=float(st.session_state.holdings.loc[st.session_state.holdings['ticker']==t,'price'].dropna().iloc[0]) if "price" in st.session_state.holdings.columns and not st.session_state.holdings.loc[st.session_state.holdings['ticker']==t,'price'].isna().all() else 0.0, key=key)
                manual_prices[t] = val if val > 0 else None

        drift_df, total_value = calculate_drift(st.session_state.holdings, st.session_state.targets, manual_prices=manual_prices)
        st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        st.metric("Available Cash Pool", f"${available_cash:,.2f}")
        st.markdown("**Current Allocation Snapshot**")
        st.dataframe(drift_df[["ticker","shares","price","current_value","current_pct","target_pct","drift_pct"]])

# -------------------------
# Action Plan Tab
# -------------------------
with tabs[1]:
    st.header("Action Plan")
    st.markdown("Recommendations are generated using the **$900 biweekly deposit** plus any **sale proceeds** from rebalancing. Buys are prioritized for underweight assets. Sells are suggested only if cash pool is insufficient to reach targets.")
    # Recompute with current session cash
    deposit_amount = BIWEEKLY_DEPOSIT if st.session_state.deposit_enabled else 0.0
    cash_pool = st.session_state.cash_pool + deposit_amount

    # Recommendation algorithm
    def generate_recommendations(drift_df, cash_pool):
        """
        Returns list of recommendations dicts:
        {'ticker','action'('BUY'/'SELL'),'shares','amount','notes'}
        Logic:
        - Compute target dollar amounts from target_pct and total portfolio value + cash_pool
        - For underweight tickers (drift_pct < 0), allocate cash_pool to buy until target reached
        - If cash_pool insufficient, consider sells from overweight tickers to free cash, but only if needed
        """
        recs = []
        df = drift_df.copy()
        total_portfolio_value = df["current_value"].sum()
        # Target dollar amounts are based on total portfolio value + cash_pool (we assume deposit will be invested)
        investable_total = total_portfolio_value + cash_pool
        df["target_value"] = df["target_pct"] / 100.0 * investable_total
        df["value_gap"] = df["target_value"] - df["current_value"]  # positive => need buy
        # Prioritize buys for negative drift (underweight)
        buys = df[df["value_gap"] > 0].sort_values("value_gap", ascending=False)
        sells = df[df["value_gap"] < 0].sort_values("value_gap")  # most overweight first (most negative gap)
        remaining_cash = cash_pool

        # First pass: allocate buys using available cash
        for _, row in buys.iterrows():
            ticker = row["ticker"]
            price = row["price"] if row["price"] > 0 else 0.0
            need = row["value_gap"]
            if price <= 0:
                # Can't compute shares without price; skip and flag
                recs.append({"ticker":ticker,"action":"BUY","shares":None,"amount":need,"notes":"Missing price - manual input required"})
                continue
            affordable = min(need, remaining_cash)
            if affordable <= 0:
                continue
            shares_to_buy = np.floor(affordable / price * 1000000) / 1000000.0  # allow fractional if broker supports; adjust precision
            if shares_to_buy > 0:
                amount = shares_to_buy * price
                recs.append({"ticker":ticker,"action":"BUY","shares":shares_to_buy,"amount":amount,"notes":f"Use cash pool ${amount:,.2f}"})
                remaining_cash -= amount

        # If after buys we still have negative gaps (i.e., buys unmet) and remaining_cash < 0.01, consider sells
        unmet_buys = buys[buys["value_gap"] > 0].copy()
        unmet_buys["remaining_need"] = unmet_buys["value_gap"]
        # compute unmet total
        unmet_total = unmet_buys["remaining_need"].sum()
        if remaining_cash < 1e-6 and unmet_total > 0:
            # Need to free cash by selling from overweight assets
            for _, row in sells.iterrows():
                ticker = row["ticker"]
                price = row["price"] if row["price"] > 0 else 0.0
                excess = -row["value_gap"]  # positive amount available to sell
                if price <= 0:
                    recs.append({"ticker":ticker,"action":"SELL","shares":None,"amount":excess,"notes":"Missing price - manual input required; tax impact may apply"})
                    continue
                # Sell only as much as needed to satisfy unmet buys
                to_sell_amount = min(excess, unmet_total)
                shares_to_sell = np.floor(to_sell_amount / price * 1000000) / 1000000.0
                if shares_to_sell > 0:
                    amount = shares_to_sell * price
                    recs.append({"ticker":ticker,"action":"SELL","shares":shares_to_sell,"amount":amount,"notes":"Sell to free cash for rebalancing; Tax-Impact: may realize gains"})
                    remaining_cash += amount
                    unmet_total -= amount
                    if unmet_total <= 0:
                        break

        # Final note: if still unmet, recommend partial buys or hold
        if remaining_cash > 0:
            recs.append({"ticker":"CASH","action":"HOLD","shares":None,"amount":remaining_cash,"notes":"Remaining cash after allocation"})
        return recs

    recs = generate_recommendations(drift_df, cash_pool)
    # Display recommendations
    rec_df = pd.DataFrame(recs)
    if rec_df.empty:
        st.info("No recommendations generated. Portfolio appears aligned with targets or missing price data.")
    else:
        st.subheader("Recommendations")
        st.dataframe(rec_df)

    # Allow user to Accept recommendations
    st.markdown("**Accept Recommendations**")
    if not rec_df.empty:
        for i, row in rec_df.iterrows():
            cols = st.columns([3,1,1,2])
            with cols[0]:
                st.write(f"**{row['action']} {row['ticker']}**")
                st.write(row.get("notes",""))
            with cols[1]:
                amt = row['amount'] if pd.notna(row['amount']) else 0.0
                st.write(f"${amt:,.2f}")
            with cols[2]:
                shares = row['shares'] if pd.notna(row['shares']) else ""
                st.write(f"{shares}")
            with cols[3]:
                key = f"accept_{i}"
                if st.button("Accept", key=key):
                    entry = {
                        "date": datetime.utcnow().isoformat(),
                        "ticker": row['ticker'],
                        "action": row['action'],
                        "amount": float(row['amount']) if pd.notna(row['amount']) else None,
                        "notes": row.get("notes","")
                    }
                    update_history(entry)
                    # Update cash pool and holdings locally
                    if row['action'] == "BUY" and pd.notna(row['amount']):
                        st.session_state.cash_pool = max(0.0, st.session_state.cash_pool - float(row['amount']) + (BIWEEKLY_DEPOSIT if st.session_state.deposit_enabled else 0.0))
                        # update holdings shares
                        if pd.notna(row['shares']):
                            t = row['ticker']
                            df = st.session_state.holdings
                            if t in df["ticker"].values:
                                df.loc[df["ticker"]==t,"shares"] += float(row['shares'])
                            else:
                                df = pd.concat([df, pd.DataFrame([{"ticker":t,"shares":float(row['shares']),"price":0.0}])], ignore_index=True)
                            st.session_state.holdings = df
                    if row['action'] == "SELL" and pd.notna(row['amount']):
                        st.session_state.cash_pool = st.session_state.cash_pool + float(row['amount'])
                        # reduce holdings shares if possible
                        if pd.notna(row['shares']):
                            t = row['ticker']
                            df = st.session_state.holdings
                            if t in df["ticker"].values:
                                df.loc[df["ticker"]==t,"shares"] = np.maximum(0.0, df.loc[df["ticker"]==t,"shares"] - float(row['shares']))
                                st.session_state.holdings = df
                    st.success(f"Accepted {row['action']} {row['ticker']}. History updated.")

    # Tax awareness note
    st.markdown("**Tax Awareness**")
    st.info("Sells suggested for rebalancing may realize capital gains or losses. This engine flags sells with a Tax-Impact note. Consult your tax advisor before executing.")

# -------------------------
# History Tab
# -------------------------
with tabs[2]:
    st.header("Recommendation History")
    st.markdown("Accepted recommendations are persisted to history.csv in the repo root.")
    # Show history with search
    hist = st.session_state.history.copy()
    q = st.text_input("Search history by ticker or notes", value="")
    if q:
        mask = hist.apply(lambda row: q.lower() in str(row.values).lower(), axis=1)
        hist_display = hist[mask]
    else:
        hist_display = hist
    st.dataframe(hist_display)

    # Allow export or manual commit note
    st.markdown("If you want to push history to GitHub, commit the updated history.csv from the repo root.")

# -------------------------
# End of App
# -------------------------
