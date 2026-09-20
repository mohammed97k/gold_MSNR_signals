#!/usr/bin/env python3
"""
Pure MSNR Strategy Bot — XAUUSD M15
Supports both LIMIT and MARKET order types (same as Pine logic)
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
LIMIT_EXPIRY_BARS = 96

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
# DATA
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
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        df["volume"] = 0.0
    return df.set_index("datetime").sort_index()

# ============================================================
# SIGNAL DETECTION
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

        is_limit = abs(c[i] - res_level) > 0.40
        order_type = "LIMIT" if is_limit else "MARKET"

        raw_dist = abs(c[i] - min(l[i], res_level)) + 0.50
        sl_dist = max(min_sl_dist, min(max_sl_dist, raw_dist))
        tp_dist = max(10.0, min(25.0, sl_dist * RR_RATIO))

        entry = res_level if is_limit else c[i]

        return {
            "dir": "BUY",
            "order_type": order_type,
            "level": res_level,
            "entry": entry,
            "sl": entry - sl_dist,
            "tp": entry + tp_dist,
            "risk_pips": sl_dist / PIP,
            "reward_pips": tp_dist / PIP,
            "signal_time": idx[i],
        }

    elif (sup_broken and sup_level is not None
          and h[i] >= sup_level >= c[i] and c[i] < o[i]):

        is_limit = abs(c[i] - sup_level) > 0.40
        order_type = "LIMIT" if is_limit else "MARKET"

        raw_dist = abs(max(h[i], sup_level) - c[i]) + 0.50
        sl_dist = max(min_sl_dist, min(max_sl_dist, raw_dist))
        tp_dist = max(10.0, min(25.0, sl_dist * RR_RATIO))

        entry = sup_level if is_limit else c[i]

        return {
            "dir": "SELL",
            "order_type": order_type,
            "level": sup_level,
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
    default = {"pending": [], "active": [], "closed": []}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            for k in default.keys():
                if k not in data:
                    data[k] = []
            return data
        except (json.JSONDecodeError, Exception) as e:
            print(f"Corrupted state file, resetting: {e}")
            return default
    return default

def save_state(state):
    state["closed"] = state["closed"][-100:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

# ============================================================
# FORMAT
# ============================================================
def format_signal(s):
    emoji = "🟢" if s["dir"] == "BUY" else "🔴"
    if s["order_type"] == "LIMIT":
        if s["dir"] == "BUY":
            order_label = "🟡 شراء معلّق (BUY LIMIT)"
        else:
            order_label = "🟡 بيع معلّق (SELL LIMIT)"
    else:
        if s["dir"] == "BUY":
            order_label = "⚡ شراء فوري (BUY MARKET)"
        else:
            order_label = "⚡ بيع فوري (SELL MARKET)"

    return (
        f"{emoji} *XAUUSD SIGNAL*\n\n"
        f"*النوع:* {order_label}\n"
        f"*الاتجاه:* *{s['dir']}*\n"
        f"*الوقت:* `{s['signal_time'].strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
        f"*دخول:*  `{s['entry']:.2f}`\n"
        f"*ستوب:*  `{s['sl']:.2f}`  ({s['risk_pips']:.1f} نقطة)\n"
        f"*هدف:*   `{s['tp']:.2f}`  ({s['reward_pips']:.1f} نقطة)\n"
        f"*R:R:*    1:{RR_RATIO}\n"
        f"*المخاطرة:* {RISK_PCT}%\n\n"
        f"_Pure MSNR v1.0_"
    )

def format_fill(sig, fill_ts):
    emoji = "🟢" if sig["dir"] == "BUY" else "🔴"
    return (
        f"✅ *LIMIT FILLED*\n\n"
        f"*الزوج:* XAUUSD\n"
        f"*الاتجاه:* {sig['dir']}\n"
        f"*دخول:* `{sig['entry']:.2f}`\n"
        f"*وقت التفعيل:* `{fill_ts.strftime('%Y-%m-%d %H:%M UTC')}`\n"
        f"*ستوب:* `{sig['sl']:.2f}`\n"
        f"*هدف:* `{sig['tp']:.2f}`"
    )

def format_expired(sig, expiry_ts):
    return (
        f"⏰ *LIMIT EXPIRED*\n\n"
        f"*الزوج:* XAUUSD\n"
        f"*الاتجاه:* {sig['dir']}\n"
        f"*مستوى:* `{sig['entry']:.2f}`\n"
        f"*انتهت الصلاحية:* `{expiry_ts.strftime('%Y-%m-%d %H:%M UTC')}`"
    )

def format_result(sig, outcome_type, exit_price, close_ts):
    is_win = outcome_type == "TP"
    emoji = "🎉" if is_win else "❌"
    pips = sig['reward_pips'] if is_win else -sig['risk_pips']
    return (
        f"{emoji} *{outcome_type} HIT*\n\n"
        f"*الزوج:* XAUUSD\n"
        f"*الاتجاه:* {sig['dir']}\n"
        f"*النوع:* {sig['order_type']}\n"
        f"*دخول:* `{sig['entry']:.2f}`\n"
        f"*خروج:*  `{exit_price:.2f}`\n"
        f"*الوقت:* `{close_ts.strftime('%Y-%m-%d %H:%M UTC')}`\n"
        f"*النتيجة:* *{pips:+.1f} نقطة*"
    )

# ============================================================
# TRACK PENDING LIMIT ORDERS
# ============================================================
async def track_pending(bot, state, df):
    still_pending = []
    for sig in state["pending"]:
        sig_time = pd.to_datetime(sig["signal_time"])
        after = df[df.index > sig_time]
        if len(after) == 0:
            still_pending.append(sig)
            continue

        filled = False
        fill_ts = None

        for ts, bar in after.iterrows():
            if sig["dir"] == "BUY":
                if bar["low"] <= sig["entry"]:
                    filled = True
                    fill_ts = ts
                    break
            else:
                if bar["high"] >= sig["entry"]:
                    filled = True
                    fill_ts = ts
                    break

        if filled:
            await send_msg(bot, format_fill(sig, fill_ts))
            sig["fill_time"] = str(fill_ts)
            state["active"].append(sig)
            print(f"  FILLED: {sig.get('id','?')}")
        else:
            bars_since = len(after)
            if bars_since >= LIMIT_EXPIRY_BARS:
                expiry_ts = after.index[-1]
                await send_msg(bot, format_expired(sig, expiry_ts))
                state["closed"].append({**sig, "outcome": "EXPIRED", "close_time": str(expiry_ts)})
                print(f"  EXPIRED: {sig.get('id','?')}")
            else:
                still_pending.append(sig)

    state["pending"] = still_pending

# ============================================================
# TRACK ACTIVE TRADES
# ============================================================
async def track_active(bot, state, df):
    still_active = []
    for sig in state["active"]:
        start_time = pd.to_datetime(sig.get("fill_time", sig["signal_time"]))
        after = df[df.index > start_time]
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

    await track_pending(bot, state, df)
    await track_active(bot, state, df)

    if len(state["pending"]) == 0 and len(state["active"]) == 0:
        sig = detect_signal(df)
        if sig:
            sig_id = f"XAU_{sig['signal_time'].strftime('%Y%m%d_%H%M')}"
            known_ids = {s.get("id") for s in state["pending"] + state["active"] + state["closed"]}

            if sig_id not in known_ids:
                sig["id"] = sig_id
                sig_time_dt = sig["signal_time"]
                sig["signal_time"] = str(sig_time_dt)

                if sig["order_type"] == "LIMIT":
                    state["pending"].append(sig)
                else:
                    state["active"].append(sig)

                await send_msg(bot, format_signal({**sig, "signal_time": sig_time_dt}))
                print(f"  NEW SIGNAL: {sig_id} ({sig['order_type']})")
            else:
                print(f"  Signal {sig_id} already known")
        else:
            print("  No signal")
    else:
        print(f"  Busy: pending={len(state['pending'])}, active={len(state['active'])}")

    save_state(state)
    print(f"Pending: {len(state['pending'])} | Active: {len(state['active'])} | Closed: {len(state['closed'])}")

if __name__ == "__main__":
    asyncio.run(main())