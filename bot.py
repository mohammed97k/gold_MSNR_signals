#!/usr/bin/env python3
"""
MSNR Two-Tier Sniper Bot
========================
A+ Sniper  → Daily anchor (High Priority)
B+ Standard → 4H anchor   (Standard)

Runs on GitHub Actions every 30 min.
"""

import os
import requests
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timezone
from telegram import Bot

# ============================================================
# CONFIG
# ============================================================
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

PAIRS = ["EURUSD", "USDJPY", "USDCAD", "AUDUSD", "NZDUSD", "EURJPY"]

PIP = {
    "EURUSD":0.0001, "USDJPY":0.01, "USDCAD":0.0001,
    "AUDUSD":0.0001, "NZDUSD":0.0001, "EURJPY":0.01,
}
MAX_SL = {
    "EURUSD":25, "USDJPY":40, "USDCAD":25,
    "AUDUSD":25, "NZDUSD":25, "EURJPY":40,
}

# A+ Sniper params
A_WICK   = 0.40
A_RETEST = 4
A_BIAS   = True

# B+ Standard params
B_WICK   = 0.35
B_RETEST = 6
B_BIAS   = False

SL_BUFFER = 4
RR        = 2.5


# ============================================================
# TELEGRAM
# ============================================================
async def send_msg(bot, text):
    try:
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=text,
            parse_mode="Markdown"
        )
        return True
    except Exception as e:
        print(f"   TELEGRAM ERROR: {e}")
        return False


# ============================================================
# DATA
# ============================================================
def fetch_recent(code, bars=500):
    symbol = f"{code[:3]}/{code[3:]}"
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "15min",
        "outputsize": bars,
        "order": "ASC",
        "timezone": "UTC",
        "apikey": TWELVE_API_KEY,
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


def resample_all(df):
    ohlc = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return {
        "M15": df,
        "H1":  df.resample("1h", label="left", closed="left").agg(ohlc).dropna(),
        "H4":  df.resample("4h", label="left", closed="left").agg(ohlc).dropna(),
        "D1":  df.resample("1D", label="left", closed="left").agg(ohlc).dropna(),
    }


# ============================================================
# MSNR LOGIC
# ============================================================
def make_levels(tf):
    o = tf["open"].values
    c = tf["close"].values
    rows = []
    for i in range(1, len(tf)):
        pb  = c[i-1] > o[i-1]
        prb = c[i-1] < o[i-1]
        cb  = c[i]   > o[i]
        crb = c[i]   < o[i]
        if pb and crb:
            rows.append({"bar":i, "price":c[i-1], "type":"A",    "dir":"SELL"})
        elif prb and cb:
            rows.append({"bar":i, "price":c[i-1], "type":"V",    "dir":"BUY"})
        elif prb and crb:
            rows.append({"bar":i, "price":c[i-1], "type":"GapS", "dir":"SELL"})
        elif pb and cb:
            rows.append({"bar":i, "price":c[i-1], "type":"GapB", "dir":"BUY"})
    return rows


def bias_ok(h4, touch_ts, direction):
    idx = h4.index.searchsorted(touch_ts, side="left")
    if idx < 5:
        return False
    lb = min(25, idx-1)
    h = h4["high"].values
    l = h4["low"].values
    c = h4["close"].values
    if direction == "BUY":
        cause = h[idx-lb:idx].max()
        return bool(np.any(c[max(0,idx-20):min(len(h4),idx+20)] > cause))
    else:
        cause = l[idx-lb:idx].min()
        return bool(np.any(c[max(0,idx-20):min(len(h4),idx+20)] < cause))


def detect_signal(code, h1, h4, anchor_tf, wick, retest_w, use_bias, grade):
    if len(anchor_tf) < 20 or len(h1) < 10:
        return None

    levels = make_levels(anchor_tf)
    if not levels:
        return None

    # Fresh filter on anchor
    a_h = anchor_tf["high"].values
    a_l = anchor_tf["low"].values
    fresh = []
    for lv in levels:
        fi = lv["bar"]
        p  = lv["price"]
        d  = lv["dir"]
        touched = False
        for j in range(fi+1, len(anchor_tf)):
            if d == "SELL" and a_h[j] >= p:
                touched = True
                break
            if d == "BUY" and a_l[j] <= p:
                touched = True
                break
        if not touched:
            fresh.append(lv)

    if not fresh:
        return None

    last = len(h1) - 2
    if last < 3:
        return None

    h_h = h1["high"].values
    h_l = h1["low"].values
    h_o = h1["open"].values
    h_c = h1["close"].values
    tol = 5 * PIP[code] if PIP[code] == 0.01 else 3 * PIP[code]

    for rej_offset in range(1, retest_w + 1):
        rej_idx = last - (rej_offset - 1)
        if rej_idx < 2:
            continue
        rng = h_h[rej_idx] - h_l[rej_idx]
        if rng <= 0:
            continue
        uw = h_h[rej_idx] - max(h_o[rej_idx], h_c[rej_idx])
        lw = min(h_o[rej_idx], h_c[rej_idx]) - h_l[rej_idx]

        for lv in fresh[-15:]:
            price = lv["price"]
            d     = lv["dir"]

            if use_bias and not bias_ok(h4, h1.index[rej_idx], d):
                continue

            if d == "BUY":
                if (h_l[rej_idx] <= price) and (h_c[rej_idx] > price) and (lw >= wick * rng):
                    rt = rej_idx + rej_offset
                    if rt > last:
                        continue
                    if h_l[rt] <= price + tol:
                        sl = h_l[rej_idx] - SL_BUFFER * PIP[code]
                        risk = price - sl
                        if risk / PIP[code] > MAX_SL[code] or risk / PIP[code] < 5:
                            continue
                        return {
                            "pair": code, "grade": grade, "dir": "BUY",
                            "level": price, "level_type": lv["type"],
                            "signal_ts": h1.index[rt],
                            "entry": price, "sl": sl,
                            "tp": price + RR * risk,
                            "risk_pips": risk / PIP[code],
                        }
            else:
                if (h_h[rej_idx] >= price) and (h_c[rej_idx] < price) and (uw >= wick * rng):
                    rt = rej_idx + rej_offset
                    if rt > last:
                        continue
                    if h_h[rt] >= price - tol:
                        sl = h_h[rej_idx] + SL_BUFFER * PIP[code]
                        risk = sl - price
                        if risk / PIP[code] > MAX_SL[code] or risk / PIP[code] < 5:
                            continue
                        return {
                            "pair": code, "grade": grade, "dir": "SELL",
                            "level": price, "level_type": lv["type"],
                            "signal_ts": h1.index[rt],
                            "entry": price, "sl": sl,
                            "tp": price - RR * risk,
                            "risk_pips": risk / PIP[code],
                        }
    return None


# ============================================================
# FORMAT
# ============================================================
def format_signal(s):
    if s["grade"] == "A+":
        header = "🎯 *MSNR A+ SNIPER* (High Priority)"
    else:
        header = "⚡ *MSNR B+ STANDARD*"

    emoji = "🟢" if s["dir"] == "BUY" else "🔴"
    p = PIP[s["pair"]]
    risk   = abs(s["entry"] - s["sl"]) / p
    reward = abs(s["tp"]   - s["entry"]) / p

    return (
        f"{header}\n\n"
        f"{emoji} *{s['pair']}* — *{s['dir']}*\n"
        f"*Type:* {s['level_type']}\n"
        f"*Signal:* `{s['signal_ts'].strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
        f"*Entry:* `{s['entry']:.5f}`\n"
        f"*SL:*    `{s['sl']:.5f}`  ({risk:.1f} pips)\n"
        f"*TP:*    `{s['tp']:.5f}`  ({reward:.1f} pips)\n"
        f"*R:R:*   1:{RR}\n\n"
        f"_Risk: 1% | MSNR v2.0_"
    )


# ============================================================
# MAIN
# ============================================================
async def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"MSNR Scan @ {now}")
    print(f"Pairs: {', '.join(PAIRS)}")

    bot = Bot(token=TELEGRAM_TOKEN)

    # Startup confirmation (each run confirms bot is alive)
    await send_msg(
        bot,
        f"✅ *MSNR Bot — Scan Started*\n\n"
        f"🕐 `{now}`\n"
        f"📊 Scanning {len(PAIRS)} pairs:\n"
        f"`{' · '.join(PAIRS)}`\n\n"
        f"_Searching for A+ and B+ setups..._"
    )

    found = 0
    signals_summary = []

    for code in PAIRS:
        try:
            m15 = fetch_recent(code, bars=500)
        except Exception as e:
            print(f"   WARN {code}: {e}")
            signals_summary.append(f"⚠️ {code}: fetch error")
            continue

        tf = resample_all(m15)
        h1, h4, d1 = tf["H1"], tf["H4"], tf["D1"]

        # Try A+ first (Daily anchor)
        sig = detect_signal(code, h1, h4, d1, A_WICK, A_RETEST, A_BIAS, "A+")

        # If no A+, try B+ (4H anchor)
        if sig is None:
            sig = detect_signal(code, h1, h4, h4, B_WICK, B_RETEST, B_BIAS, "B+")

        if sig:
            await send_msg(bot, format_signal(sig))
            print(f"   SIGNAL: {sig['grade']} {code} {sig['dir']} @ {sig['level']:.5f}")
            signals_summary.append(f"🎯 {code} — {sig['grade']} {sig['dir']}")
            found += 1
        else:
            print(f"   --     {code}: no signal")
            signals_summary.append(f"—  {code}: no signal")

    # End of scan summary
    end_ts = datetime.now(timezone.utc).strftime("%H:%M UTC")
    summary = "\n".join(signals_summary)

    if found == 0:
        await send_msg(
            bot,
            f"🔍 *Scan Complete — No Setups*\n\n"
            f"🕐 `{end_ts}`\n\n"
            f"{summary}\n\n"
            f"_Next scan in 30 min_"
        )
    else:
        await send_msg(
            bot,
            f"📊 *Scan Complete — {found} Signal(s)*\n\n"
            f"🕐 `{end_ts}`\n\n"
            f"{summary}\n\n"
            f"_Check signals above ☝️_"
        )

    print(f"\nDone. {found} signal(s).")


if __name__ == "__main__":
    asyncio.run(main())