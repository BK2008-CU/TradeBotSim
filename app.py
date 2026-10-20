"""Streamlit UI for the Alpaca signal bot.  Run with:  streamlit run app.py"""

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import market_data as md
import signals as sg
from broker import Broker
from universe import UNIVERSE, all_tickers

load_dotenv()
st.set_page_config(page_title="Signal Bot", page_icon="📈", layout="wide")

PAPER = os.getenv("ALPACA_PAPER", "true").lower() != "false"
COLORS = {"BUY": "#1b9e5a", "HOLD": "#8a8f98", "SELL": "#d6453d"}

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📈 Signal Bot")
    st.caption("Paper trading" if PAPER else "⚠️ LIVE trading enabled via ALPACA_PAPER=false")

    key = st.text_input("Alpaca API key", value=os.getenv("ALPACA_API_KEY", ""), type="password")
    secret = st.text_input("Alpaca secret", value=os.getenv("ALPACA_SECRET_KEY", ""), type="password")

    st.divider()
    st.subheader("Universe")
    sectors = st.multiselect("Sectors", list(UNIVERSE), default=["Mega-cap Tech", "Semiconductors"])
    picked = sorted({t for s in sectors for t in UNIVERSE[s]})
    extra = st.text_input("Extra tickers (comma separated)", placeholder="e.g. UBER, SHOP")
    tickers = sorted(set(picked) | {t.strip().upper() for t in extra.split(",") if t.strip()})
    st.caption(f"{len(tickers)} tickers selected")

    st.divider()
    st.subheader("Signal weights")
    w_tech = st.slider("Technical", 0.0, 1.0, 0.4, 0.05)
    w_ml = st.slider("ML model", 0.0, 1.0, 0.3, 0.05)
    w_sent = st.slider("News sentiment", 0.0, 1.0, 0.3, 0.05)
    buy_th = st.slider("BUY above", 0.05, 0.8, 0.25, 0.05)
    sell_th = st.slider("SELL below", -0.8, -0.05, -0.25, 0.05)
    use_news = st.checkbox("Fetch news (slower)", value=True)

weights = {"technical": w_tech, "ml": w_ml, "sentiment": w_sent}

st.title("Stock signal dashboard")
st.caption(
    "Decision-support tool for paper trading. Signals are not financial advice, "
    "and short-term price prediction is unreliable."
)

if not key or not secret:
    st.info("Enter your Alpaca paper-trading API keys in the sidebar (or put them in a .env file).")
    st.stop()


# ---------------------------------------------------------------- scanning
@st.cache_data(ttl=900, show_spinner=False)
def load_bars(key, secret, symbols: tuple):
    return md.fetch_bars(key, secret, list(symbols))


@st.cache_data(ttl=900, show_spinner=False)
def load_news(key, secret, symbol: str):
    return md.fetch_headlines(key, secret, symbol)


def run_scan():
    with st.spinner("Downloading price history..."):
        bars = load_bars(key, secret, tuple(tickers))
    results, news = {}, {}
    bar = st.progress(0.0, text="Analyzing...")
    for i, sym in enumerate(tickers):
        if sym in bars and len(bars[sym]) > 80:
            news[sym] = load_news(key, secret, sym) if use_news else []
            results[sym] = sg.analyze(bars[sym], news[sym], weights, buy_th, sell_th)
        bar.progress((i + 1) / len(tickers), text=f"Analyzing {sym}")
    bar.empty()
    st.session_state["scan"] = {"results": results, "news": news, "missing": [t for t in tickers if t not in results]}


if st.button("🔍 Run scan", type="primary", disabled=not tickers):
    try:
        run_scan()
    except Exception as e:
        st.error(f"Scan failed: {e}")

scan = st.session_state.get("scan")
tab_scan, tab_detail, tab_port = st.tabs(["Scanner", "Deep dive", "Portfolio"])

# ---------------------------------------------------------------- scanner tab
with tab_scan:
    if not scan or not scan["results"]:
        st.write("Pick your universe in the sidebar, then click **Run scan**.")
    else:
        rows = [
            {
                "Ticker": s, "Signal": r["signal"], "Score": r["score"], "Price": r["price"],
                "1d %": r["change_1d"] * 100, "Technical": r["technical"], "ML": r["ml"],
                "Sentiment": r["sentiment"], "ML acc.": r["ml_accuracy"],
            }
            for s, r in scan["results"].items()
        ]
        df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("BUY signals", int((df["Signal"] == "BUY").sum()))
        c2.metric("HOLD signals", int((df["Signal"] == "HOLD").sum()))
        c3.metric("SELL signals", int((df["Signal"] == "SELL").sum()))

        filt = st.multiselect("Show signals", ["BUY", "HOLD", "SELL"], default=["BUY", "HOLD", "SELL"])
        view = df[df["Signal"].isin(filt)]
        styled = view.style.map(
            lambda v: f"color: {COLORS.get(v, 'inherit')}; font-weight: 600", subset=["Signal"]
        ).format({"Price": "${:,.2f}", "1d %": "{:+.2f}%", "Score": "{:+.2f}", "Technical": "{:+.2f}",
                  "ML": "{:+.2f}", "Sentiment": "{:+.2f}", "ML acc.": "{:.0%}"}, na_rep="–")
        st.dataframe(styled, use_container_width=True, hide_index=True, height=520)
        if scan["missing"]:
            st.caption(f"No usable data for: {', '.join(scan['missing'])}")

# ---------------------------------------------------------------- deep dive tab
with tab_detail:
    if not scan or not scan["results"]:
        st.write("Run a scan first.")
    else:
        sym = st.selectbox("Ticker", list(scan["results"]))
        r = scan["results"][sym]
        d = r["indicators"].tail(180)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Signal", r["signal"])
        m2.metric("Combined score", f"{r['score']:+.2f}")
        m3.metric("Price", f"${r['price']:,.2f}", f"{r['change_1d'] * 100:+.2f}%")
        m4.metric("ML holdout accuracy", "–" if r["ml_accuracy"] is None else f"{r['ml_accuracy']:.0%}")

        fig = go.Figure()
        fig.add_candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Price")
        fig.add_scatter(x=d.index, y=d["sma20"], name="SMA 20", line=dict(width=1.2))
        fig.add_scatter(x=d.index, y=d["sma50"], name="SMA 50", line=dict(width=1.2))
        fig.add_scatter(x=d.index, y=d["bb_upper"], name="Bollinger", line=dict(width=0.8, dash="dot", color="#999"))
        fig.add_scatter(x=d.index, y=d["bb_lower"], showlegend=False, line=dict(width=0.8, dash="dot", color="#999"))
        fig.update_layout(height=430, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.subheader("What's driving the signal")
            parts = {**r["components"], "ML model": r["ml"], "News sentiment": r["sentiment"]}
            st.bar_chart(pd.Series(parts, name="score"))
            st.caption("Positive = bullish, negative = bearish. Each bar is on a -1 to +1 scale.")
        with right:
            st.subheader("Recent headlines")
            heads = scan["news"].get(sym, [])
            if not heads:
                st.write("No headlines loaded (news may be turned off).")
            for h in heads[:8]:
                s = sg.sentiment_score([h])
                st.write(f"{'🟢' if s > 0.1 else '🔴' if s < -0.1 else '⚪'} {h}")

        st.divider()
        st.subheader("Paper order")
        dollars = st.number_input("Dollar amount to buy", min_value=1.0, value=100.0, step=25.0)
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button(f"Buy ${dollars:,.0f} of {sym}"):
            try:
                o = Broker(key, secret, PAPER).buy_dollars(sym, dollars)
                st.success(f"Order submitted: {o.symbol} ({o.status})")
            except Exception as e:
                st.error(f"Order failed: {e}")
        if b2.button(f"Close {sym} position"):
            try:
                Broker(key, secret, PAPER).close(sym)
                st.success(f"Close order submitted for {sym}")
            except Exception as e:
                st.error(f"Close failed: {e}")

# ---------------------------------------------------------------- portfolio tab
with tab_port:
    try:
        broker = Broker(key, secret, PAPER)
        acct = broker.account()
        a1, a2, a3 = st.columns(3)
        a1.metric("Equity", f"${acct['equity']:,.2f}")
        a2.metric("Cash", f"${acct['cash']:,.2f}")
        a3.metric("Buying power", f"${acct['buying_power']:,.2f}")

        pos = broker.positions()
        if pos:
            pdf = pd.DataFrame(pos).rename(columns={
                "symbol": "Ticker", "qty": "Shares", "avg_entry": "Avg entry",
                "market_value": "Market value", "unrealized_pl": "P/L ($)", "unrealized_plpc": "P/L (%)",
            })
            st.dataframe(
                pdf.style.format({"Shares": "{:.4f}", "Avg entry": "${:,.2f}", "Market value": "${:,.2f}",
                                  "P/L ($)": "${:+,.2f}", "P/L (%)": "{:+.2%}"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.write("No open positions.")
        st.button("Refresh")
    except Exception as e:
        st.error(f"Could not reach Alpaca: {e}")