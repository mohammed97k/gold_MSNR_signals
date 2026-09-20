#!/usr/bin/env python3
"""
Pure MSNR Strategy Bot — XAUUSD M15
Strategy: Pivot BOS + Session + SL/TP 1:1.8
No Bias, No Story, No Filters (Original)
"""
import os, requests, pandas as pd, numpy as np, asyncio, json
from datetime import datetime, timezone
from telegram import Bot

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

# ============================================================
# STRATEGY PARAMETERS
# ============================================================
PAIR = "XAU/USD"
PIP = 0.10
SL_PTS_MIN = 50.0
SL_PTS_MAX = 130.0
RR_RATIO   = 1.8
LEN_PIVOT  = 10
SESSION_START = 7
SESSION_END   = 18
RISK_PCT      = 1.0

STATE_FILE = "state.json"

# ============================================================
# TELEGRAM
# ============================================================
async def send_msg(bot, text):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        return True
    except Exception as e:
        print(f"TG error: {e}")
        return False

# ============================================================
# DATA FETCH
# ============================================================
def fetch_m15(bars=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": PAIR, "interval": "15min", "outputsize": bars,
        "order": "ASC", "timezone": "UTC", "apikey": TWELVE_API_KEY
    }
    r = requests.get(url, params=params, timeout=20).json()
    if "values" not in r:
        raise RuntimeError(f"Twelve error: {r}")
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c])
    df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
    return df.set_index("datetime").sort_index()

# ============================================================
# STRATEGY LOGIC
# ============================================================
def detect_signal(df):
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    o = df['open'].values
    idx = df.index
    n = len(df)

    if n < 100:
        return None

    win = LEN_PIVOT * 2 + 1
    roll_high = df['high'].rolling(win, center=True).max()
    roll_low  = df['low'].rolling(win, center=True).min()
    ph = np.where(df['high'].values == roll_high.values, df['high'].values, np.nan)
    pl = np.where(df['low'].values  == roll_low.values,  df['low'].values,  np.nan)
    ph_conf = pd.Series(ph).shift(LEN_PIVOT).values
    pl_conf = pd.Series(pl).shift(LEN_PIVOT).values

    res_level = None
    sup_level = None
    res_broken = False
    sup_broken = False

    for i in range(50, n):
        if not np.isnan(ph_conf[i]):
            res_level = ph_conf[i]
            res_broken = False
        if not np.isnan(pl_conf[i]):
            sup_level = pl_conf[i]
            sup_broken = False

        if res_level is not None and not res_broken and c[i] > res_level:
            res_broken = True
        if sup_level is not None and not sup_broken and c[i] < sup_level:
            sup_broken = True

    i = n - 2

    if idx[i].hour < SESSION_START or idx[i].hour >= SESSION_END:
        return None

    min_sl_dist = SL_PTS_MIN * PIP
    max_sl_dist = SL_PTS_MAX * PIP

    if (res_broken and res_level is not None
        and l[i] <= res_level <= c[i] and c[i] > o[i]):
        entry = c[i]
        raw_dist = abs(c[i] - min(l[i], res_level)) + 0.50
        sl_dist = max(min_sl_dist, min(max_sl_dist, raw_dist))
        tp_dist = max(10.0, min(25.0, sl_dist * RR_RATIO))
        return {
            "dir": "BUY",
            "entry": entry,
            "sl": entry - sl_dist,
            "tp": entry + tp_dist,
            "risk_pips": sl_dist / PIP,
            "reward_pips": tp_dist / PIP,
            "signal_time": idx[i],
        }

    elif (sup_broken and sup_level is not None
          and h[i] >= sup_level >= c[i] and c[i] < o[i]):
        entry = c[i]
        raw_dist = abs(max(h[i], sup_level) - c[i]) + 0.50
        sl_dist = max(min_sl_dist, min(max_sl_dist, raw_dist))
        tp_dist = max(10.0, min(25.0, sl_dist * RR_RATIO))
        return {
            "dir": "SELL",
            "entry": entry,
            "sl": entry + sl_dist,
            "tp": entry - tp_dist,
            "risk_pips": sl_dist / PIP,
            "reward_pips": tp_dist / PIP,
            "signal_time": idx[i],
        }

    return None

# ============================================================
# STATE
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"active": [], "closed": []}

def save_state(state):
    state["closed"] = state["closed"][-100:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ============================================================
# FORMAT
# ============================================================
def format_signal(s):
    emoji = "🟢" if s["dir"] == "BUY" else "🔴"
    return (
        f"{emoji} *XAUUSD SIGNAL*\n\n"
        f"*Direction:* *{s['dir']}*\n"
        f"*Time:* `{s['signal_time'].strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
        f"*Entry:*  `{s['entry']:.2f}`\n"
        f"*SL:*     `{s['sl']:.2f}`  ({s['risk_pips']:.1f} pips)\n"
        f"*TP:*     `{s['tp']:.2f}`  ({s['reward_pips']:.1f} pips)\n"
        f"*R:R:*    1:{RR_RATIO}\n"
        f"*Risk:*   {RISK_PCT}%\n\n"
        f"_Pure MSNR v1.0_"
    )

def format_result(sig, outcome_type, exit_price, close_ts):
    is_win = outcome_type == "TP"
    emoji = "🎉" if is_win else "❌"
    pips = sig['reward_pips'] if is_win else -sig['risk_pips']
    return (
        f"{emoji} *{outcome_type} HIT*\n\n"
        f"*Pair:* XAUUSD\n"
        f"*Dir:* {sig['dir']}\n"
        f"*Entry:* `{sig['entry']:.2f}`\n"
        f"*Exit:*  `{exit_price:.2f}`\n"
        f"*Time:* `{close_ts.strftime('%Y-%m-%d %H:%M UTC')}`\n"
        f"*Result:* *{pips:+.1f} pips*"
    )

# ============================================================
# TRACK ACTIVE TRADES
# ============================================================
async def track_active(bot, state, df):
    still_active = []
    for sig in state["active"]:
        sig_time = pd.to_datetime(sig["signal_time"])
        after = df[df.index > sig_time]
        if len(after) == 0:
            still_active.append(sig)
            continue

        outcome = None
        for ts, bar in after.iterrows():
            if sig["dir"] == "BUY":
                if bar["low"] <= sig["sl"]:
                    outcome = ("SL", sig["sl"], ts); break
                if bar["high"] >= sig["tp"]:
                    outcome = ("TP", sig["tp"], ts); break
            else:
                if bar["high"] >= sig["sl"]:
                    outcome = ("SL", sig["sl"], ts); break
                if bar["low"] <= sig["tp"]:
                    outcome = ("TP", sig["tp"], ts); break

        if outcome:
            await send_msg(bot, format_result(sig, outcome[0], outcome[1], outcome[2]))
            state["closed"].append({**sig, "outcome": outcome[0], "close_time": str(outcome[2])})
            print(f"  CLOSED: {sig.get('id','?')} -> {outcome[0]}")
        else:
            still_active.append(sig)

    state["active"] = still_active

# ============================================================
# MAIN
# ============================================================
async def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Scan @ {now}")
    bot = Bot(token=TELEGRAM_TOKEN)
    state = load_state()

    try:
        df = fetch_m15(500)
    except Exception as e:
        print(f"Fetch error: {e}")
        return

    await track_active(bot, state, df)

    sig = detect_signal(df)
    if sig:
        sig_id = f"XAU_{sig['signal_time'].strftime('%Y%m%d_%H%M')}"
        active_ids = {s.get("id") for s in state["active"]}
        closed_ids = {s.get("id") for s in state["closed"]}

        if sig_id not in active_ids and sig_id not in closed_ids:
            sig["id"] = sig_id
            sig_time_dt = sig["signal_time"]
            sig["signal_time"] = str(sig_time_dt)
            state["active"].append(sig)
            await send_msg(bot, format_signal({**sig, "signal_time": sig_time_dt}))
            print(f"  NEW SIGNAL: {sig_id}")
        else:
            print(f"  Signal {sig_id} already known")
    else:
        print("  No signal")

    save_state(state)
    print(f"Active: {len(state['active'])} | Closed: {len(state['closed'])}")

if __name__ == "__main__":
    asyncio.run(main())