"""
Bull Flag Scanner — QuantGaps Research
Standalone Streamlit app — v1.0 production
SL=2% · TP=13% · MH=15 · SMA50 · No vol filters
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Bull Flag Scanner | QuantGaps",
    page_icon="🚩",
    layout="wide",
)

# ── Production params (locked v1.0) ───────────────────
SL            = 0.03
TP            = 0.20
MAX_HOLD      = 18
NOTIONAL      = 2000.0
MAX_POSITIONS = 10
TRADING_DAYS  = 252
PERIOD        = "5y"
BATCH_SIZE    = 20

# Detection constants
POLE_MIN_BARS  = 4;   POLE_MAX_BARS  = 14
POLE_MIN_PCT   = 0.06; POLE_MAX_PCT  = 0.26
FLAG_MIN_BARS  = 2;   FLAG_MAX_BARS  = 8
MAX_RETRACE    = 0.35; MIN_RETRACE  = 0.05
MAX_FLAG_RANGE = 0.40
BRK_BUFFER     = 0.003

# ── Universe ──────────────────────────────────────────
TICKERS = [
    "SPY","QQQ","DIA","IWM","VTI","VOO","SMH",
    "XLF","XLE","XLK","XLV","XLI","XLP","XLY","XLU",
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA",
    "AVGO","ADBE","CRM","NFLX","AMD","INTC","ORCL","CSCO",
    "QCOM","TXN","AMAT","NOW","TSM","ASML","KLAC","LRCX",
    "NXPI","ON","MU",
    "BRK-B","JPM","V","MA","GS","MS","BAC","WFC","BLK","C",
    "UNH","LLY","TMO","ISRG","PFE","JNJ","BMY","REGN",
    "VRTX","MRNA","ABBV","MRK",
    "HD","COST","PG","KO","PEP","WMT","DIS","MCD","NKE",
    "SBUX","LOW","TGT","BKNG","ABNB","MDLZ","PM","BTI","CL","GIS","KMB",
    "LIN","NEE","RTX","HON","CAT","DE","LMT","BA","SLB","COP","XOM","CVX",
    "SHOP","SNOW","PLTR","CRWD","PANW","ZS","DDOG","MDB","COIN","RIVN",
]

# ── Detection ─────────────────────────────────────────
def detect_bull_flag(df, di):
    """
    di = breakout bar.
    Flag window = [di-flag_bars .. di-1] — does NOT include di.
    Returns dict with flag_resist or None.
    """
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values

    if di < POLE_MAX_BARS + FLAG_MAX_BARS + 52 or di >= len(close) - 1:
        return None
    if close[di] < np.mean(close[di-50:di]):
        return None

    for flag_bars in range(FLAG_MIN_BARS, FLAG_MAX_BARS + 1):
        flag_start  = di - flag_bars
        flag_end    = di - 1
        if flag_start < POLE_MAX_BARS + 2:
            continue
        flag_resist = float(np.max(high[flag_start:flag_end + 1]))
        flag_low    = float(np.min(low[flag_start:flag_end + 1]))

        # Fresh breakout: previous close below, current close above
        if float(close[di - 1]) > flag_resist:
            continue
        if float(close[di]) < flag_resist * (1 + BRK_BUFFER):
            continue

        pole_end_i = flag_start - 1
        for pole_bars in range(POLE_MIN_BARS, POLE_MAX_BARS + 1):
            pole_start_i = pole_end_i - pole_bars + 1
            if pole_start_i < 1:
                continue
            pole_low  = float(np.min(low[pole_start_i:pole_end_i + 1]))
            pole_high = float(high[pole_end_i])
            pole_gain = (pole_high - pole_low) / pole_low
            if not (POLE_MIN_PCT <= pole_gain <= POLE_MAX_PCT):
                continue
            pole_range = pole_high - pole_low
            retrace = (pole_high - flag_low) / pole_range
            if not (MIN_RETRACE <= retrace <= MAX_RETRACE):
                continue
            if (flag_resist - flag_low) / pole_range > MAX_FLAG_RANGE:
                continue
            if flag_resist > pole_high * 1.02:
                continue
            return {
                'flag_resist':    round(flag_resist, 2),
                'pole_gain_pct':  round(pole_gain * 100, 1),
                'pole_bars':      pole_bars,
                'flag_bars':      flag_bars,
                'retrace_pct':    round(retrace * 100, 1),
            }
    return None

# ── Data download ─────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(universe):
    batches = [universe[i:i+BATCH_SIZE] for i in range(0, len(universe), BATCH_SIZE)]
    data = {}
    for batch in batches:
        try:
            raw = yf.download(batch, period=PERIOD, interval="1d",
                              group_by="ticker", progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                continue
            for t in batch:
                try:
                    if hasattr(raw.columns, "levels") and len(raw.columns.levels) > 1:
                        if t not in raw.columns.get_level_values(0):
                            continue
                        df = raw[t].copy()
                    else:
                        df = raw.copy()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0] for c in df.columns]
                    df = df.dropna()
                    if not {"Open","High","Low","Close"}.issubset(df.columns):
                        continue
                    df.index = pd.to_datetime(df.index)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    df = df.sort_index()
                    if len(df) < 250:
                        continue
                    data[t] = df[["Open","High","Low","Close","Volume"]] \
                        if "Volume" in df.columns else df[["Open","High","Low","Close"]]
                except Exception:
                    continue
        except Exception:
            continue
    return data

# ── Live signals ──────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def find_live_signals(_data):
    cutoff = pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(5)
    results = []
    for ticker, df in _data.items():
        for di in range(max(1, len(df) - 10), len(df)):
            if df.index[di] < cutoff:
                continue
            sig = detect_bull_flag(df, di)
            if sig is None:
                continue
            ct = float(df["Close"].iloc[di])
            bl = sig["flag_resist"]
            results.append({
                "Ticker":        ticker,
                "Date":          df.index[di].strftime("%Y-%m-%d"),
                "Close":         round(ct, 2),
                "Flag Resist":   bl,
                "Pole Gain%":    sig["pole_gain_pct"],
                "Pole Bars":     sig["pole_bars"],
                "Flag Bars":     sig["flag_bars"],
                "Retrace%":      sig["retrace_pct"],
                "Entry (next open)": "next open > " + str(bl),
                "SL Price":      round(ct * (1 - SL), 2),
                "TP Price":      round(ct * (1 + TP), 2),
            })
    results.sort(key=lambda x: x["Pole Gain%"], reverse=True)
    return results

# ── Backtest ──────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(_data):
    # Build date index
    date_set = set()
    for df in _data.values():
        date_set.update(df.index.tolist())
    dates = sorted(date_set)

    # Pre-compute signals
    signals = {}
    for ticker, df in _data.items():
        tsigs = {}
        for di in range(1, len(df)):
            date = df.index[di]
            if date not in date_set:
                continue
            sig = detect_bull_flag(df, di)
            if sig is None:
                continue
            bl = sig["flag_resist"]
            ct = float(df["Close"].iloc[di])
            cy = float(df["Close"].iloc[di - 1])
            # confirm fresh cross: prev close ≤ bl < curr close
            if not (cy <= bl < ct):
                continue
            if ct < bl * (1 + BRK_BUFFER):
                continue
            tsigs[date] = sig
        if tsigs:
            signals[ticker] = tsigs

    # Simulate
    open_pos  = {}
    pending   = {}
    trades    = []
    daily_pnl = np.zeros(len(dates))

    for di in range(1, len(dates)):
        date = dates[di]

        # Enter pending positions (open-confirmation)
        if date in pending:
            cands = sorted(pending.pop(date),
                           key=lambda x: x[1], reverse=True)  # sort by pole_gain desc
            for ticker, pole_gain, signal in cands:
                if len(open_pos) >= MAX_POSITIONS:
                    break
                if ticker in open_pos:
                    continue
                df = _data.get(ticker)
                if df is None or date not in df.index:
                    continue
                o = float(df.loc[date, "Open"])
                if not np.isfinite(o) or o <= 0:
                    continue
                # Open-confirmation: open must be above flag_resist
                if o < signal["flag_resist"]:
                    continue
                open_pos[ticker] = dict(
                    entry_price=o, shares=NOTIONAL / o,
                    sl_price=o * (1 - SL), tp_price=o * (1 + TP),
                    days_held=0, entry_di=di, entry_date=date,
                    pole_gain=signal["pole_gain_pct"],
                )

        # Manage open positions
        day_pnl = 0.0
        to_close = []
        for ticker, pos in open_pos.items():
            df = _data.get(ticker)
            if df is None or date not in df.index:
                continue
            bar   = df.loc[date]
            lo    = float(bar["Low"])
            hi    = float(bar["High"])
            cl    = float(bar["Close"])
            pos["days_held"] += 1
            reason = ep = None
            if   np.isfinite(lo) and lo <= pos["sl_price"]: reason, ep = "SL", pos["sl_price"]
            elif np.isfinite(hi) and hi >= pos["tp_price"]: reason, ep = "TP", pos["tp_price"]
            elif pos["days_held"] >= MAX_HOLD:               reason, ep = "MH", cl
            if reason:
                pnl = (ep - pos["entry_price"]) * pos["shares"]
                day_pnl += pnl
                trades.append({
                    "ticker":       ticker,
                    "pnl":          round(pnl, 2),
                    "reason":       reason,
                    "hold":         di - pos["entry_di"],
                    "entry_date":   pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date":    date.strftime("%Y-%m-%d"),
                    "entry_price":  round(pos["entry_price"], 2),
                    "exit_price":   round(ep, 2),
                    "pole_gain%":   pos["pole_gain"],
                })
                to_close.append(ticker)

        daily_pnl[di] = day_pnl
        for t in to_close:
            open_pos.pop(t, None)

        # Queue next-day entries
        if di < len(dates) - 1:
            next_date = dates[di + 1]
            for ticker, tsigs in signals.items():
                if ticker in open_pos or date not in tsigs:
                    continue
                sig = tsigs[date]
                pending.setdefault(next_date, []).append(
                    (ticker, sig["pole_gain_pct"], sig))

    return trades, daily_pnl, len(dates)


def calc_metrics(trades, daily_pnl, n_dates):
    if not trades:
        return {}
    pnls    = np.array([t["pnl"] for t in trades])
    reasons = [t["reason"] for t in trades]
    holds   = np.array([t["hold"] for t in trades], dtype=float)
    n       = len(trades)
    wins    = int((pnls > 0).sum())
    capital = NOTIONAL * MAX_POSITIONS
    n_years = n_dates / TRADING_DAYS
    total   = float(pnls.sum())
    cagr    = ((1 + total / capital) ** (1 / n_years) - 1) * 100 if capital and n_years else 0
    cum     = np.cumsum(daily_pnl)
    std     = daily_pnl.std()
    sharpe  = (daily_pnl.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0
    peak    = np.maximum.accumulate(cum)
    max_dd  = float((cum - peak).min())
    calmar  = cagr / abs(max_dd / capital * 100) if max_dd else 0
    return dict(
        n=n, wr=round(wins / n * 100, 1), total=round(total, 2),
        cagr=round(cagr, 2), sharpe=round(sharpe, 3),
        calmar=round(calmar, 3), max_dd=round(max_dd, 2),
        avg_hold=round(float(holds.mean()), 1),
        pct_sl=round(reasons.count("SL") / n * 100, 1),
        pct_tp=round(reasons.count("TP") / n * 100, 1),
        pct_mh=round(reasons.count("MH") / n * 100, 1),
        cum_pnl=cum,
    )


# ══════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════
st.title("🚩 Bull Flag Scanner")
st.caption(
    f"QuantGaps Research · v1.0 · "
    f"SL {int(SL*100)}% · TP {int(TP*100)}% · MaxHold {MAX_HOLD} · SMA50 · "
    f"{len(TICKERS)} tickers"
)

tab1, tab2 = st.tabs(["🟢 LIVE SIGNALS", "🔵 BACKTEST"])

# ── TAB 1: LIVE SIGNALS ───────────────────────────────
with tab1:
    st.subheader("Live Bull Flag Breakouts — last 5 trading days")
    st.caption("Entry: next-day open above flag resistance | SL=2% | TP=13% | MaxHold=15")

    if st.button("▶ Run Live Scan", type="primary", key="live"):
        with st.spinner("Downloading data..."):
            data = download_data(tuple(TICKERS))
        with st.spinner("Scanning for Bull Flag breakouts..."):
            live_sigs = find_live_signals(data)

        if not live_sigs:
            st.info("No Bull Flag breakouts detected in the last 5 trading days.")
        else:
            st.success(f"✅ {len(live_sigs)} signal(s) found")
            df_sig = pd.DataFrame(live_sigs)
            st.dataframe(df_sig.astype(str), use_container_width=True, hide_index=True)
            st.caption("⚠️ Entry next trading day at open, only if open > Flag Resist")

            top = live_sigs[0]
            st.divider()
            st.markdown(f"### 🏆 Top Signal: **{top['Ticker']}**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Close",        f"${top['Close']}")
            c2.metric("Flag Resist",  f"${top['Flag Resist']}")
            c3.metric("Pole Gain",    f"{top['Pole Gain%']}%")
            c4.metric("SL",           f"${top['SL Price']}", f"-{int(SL*100)}%", delta_color="inverse")
            c5.metric("TP",           f"${top['TP Price']}", f"+{int(TP*100)}%")

# ── TAB 2: BACKTEST ───────────────────────────────────
with tab2:
    st.subheader("5-Year Backtest · Bull Flag v1.0")
    st.caption("Reference only — historical simulation, not forward-looking")

    if st.button("▶ Run Backtest", type="primary", key="bt"):
        with st.spinner("Downloading 5y data..."):
            data = download_data(tuple(TICKERS))
        with st.spinner("Running backtest..."):
            trades, daily_pnl, n_dates = run_backtest(data)
            m = calc_metrics(trades, daily_pnl, n_dates)

        if not m:
            st.warning("No trades generated.")
        else:
            st.divider()
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Trades",   m["n"])
            c2.metric("Win Rate", f"{m['wr']}%")
            c3.metric("CAGR",     f"{m['cagr']}%")
            c4.metric("Sharpe",   m["sharpe"])
            c5.metric("Calmar",   m["calmar"])
            c6.metric("Max DD",   f"${m['max_dd']:,.0f}")

            c1b,c2b,c3b,c4b = st.columns(4)
            c1b.metric("Total P&L",  f"${m['total']:,.0f}")
            c2b.metric("Avg Hold",   f"{m['avg_hold']} days")
            c3b.metric("SL exits",   f"{m['pct_sl']}%")
            c4b.metric("TP exits",   f"{m['pct_tp']}%")

            st.divider()
            st.markdown("#### Equity Curve")
            cum_df = pd.DataFrame({"Cumulative P&L ($)": m["cum_pnl"]})
            st.line_chart(cum_df, use_container_width=True)

            st.divider()
            st.markdown("#### Trade Log")
            df_trades = pd.DataFrame(trades)
            df_trades["pnl"] = df_trades["pnl"].round(2)
            df_trades.columns = [c.replace("_", " ").title() for c in df_trades.columns]
            df_trades = df_trades.sort_values("Exit Date", ascending=False)
            st.dataframe(df_trades.astype(str), use_container_width=True, hide_index=True)
