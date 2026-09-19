#!/usr/bin/env python3
"""
MSNR Bot — Auto-scan for A+ setups
Runs on GitHub Actions every 30 min
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import asyncio
from telegram import Bot

# ==== CONFIG FROM ENVIRONMENT ====
TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

PAIRS = ["EURUSD", "USDJPY", "USDCAD", "AUDUSD"]
PIP = {"EURUSD":0.0001,"USDJPY":0.01,"USDCAD":0.0001,"AUDUSD":0.0001}


def fetch_recent(code, bars=500):
    symbol = f"{code[:3]}/{code[3:]}"
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol":symbol,"interval":"15min","outputsize":bars,
              "order":"ASC","timezone":"UTC","apikey":TWELVE_API_KEY}
    r = requests.get(url, params=params, timeout=20).json()
    if "values" not in r:
        raise RuntimeError(f"Twelve: {r}")
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c])
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        df["volume"] = 0.0
    return df.set_index("datetime").sort_index()


def resample_all(df):
    ohlc = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    res = {"M15": df}
    for tf, rule in [("H1","1h"),("H4","4h"),("D1","1D")]:
        res[tf] = df.resample(rule, label="left", closed="left").agg(ohlc).dropna()
    return res


def detect_live_signal(code):
    try:
        m15 = fetch_recent(code, bars=500)
    except Exception as e:
        print(f"   ⚠ {code}: {e}"); return None

    tf = resample_all(m15)
    d1, h1 = tf["D1"], tf["H1"]
    if len(d1) < 20 or len(h1) < 10: return None

    o = d1["open"].values; c = d1["close"].values
    dly = []
    for i in range(1, len(d1)):
        pb = c[i-1]>o[i-1]; prb = c[i-1]<o[i-1]
        cb = c[i]>o[i]; crb = c[i]<o[i]
        if pb and crb: dly.append({"bar":i,"price":c[i-1],"type":"A","dir":"SELL"})
        elif prb and cb: dly.append({"bar":i,"price":c[i-1],"type":"V","dir":"BUY"})
        elif prb and crb: dly.append({"bar":i,"price":c[i-1],"type":"GapS","dir":"SELL"})
        elif pb and cb: dly.append({"bar":i,"price":c[i-1],"type":"GapB","dir":"BUY"})
    if not dly: return None

    last = len(h1) - 2
    if last < 3: return None

    h_h = h1["high"].values; h_l = h1["low"].values
    h_o = h1["open"].values; h_c = h1["close"].values
    tol = 5*PIP[code] if PIP[code]==0.01 else 3*PIP[code]

    for rej_offset in [1, 2]:
        rej_idx = last - (rej_offset - 1)
        if rej_idx < 2: continue
        rng = h_h[rej_idx] - h_l[rej_idx]
        if rng <= 0: continue
        uw = h_h[rej_idx] - max(h_o[rej_idx], h_c[rej_idx])
        lw = min(h_o[rej_idx], h_c[rej_idx]) - h_l[rej_idx]

        for lv in dly[-20:]:
            price = lv["price"]; d = lv["dir"]
            if d == "BUY":
                if (h_l[rej_idx]<=price) and (h_c[rej_idx]>price) and (lw>=0.5*rng):
                    retest_idx = rej_idx + rej_offset
                    if retest_idx > last: continue
                    if h_l[retest_idx] <= price + tol:
                        sl = h_l[rej_idx] - 4*PIP[code]
                        risk = price - sl
                        return {"pair":code,"dir":"BUY","level":price,
                                "level_type":lv["type"],"rej_ts":h1.index[rej_idx],
                                "signal_ts":h1.index[retest_idx],
                                "entry":price,"sl":sl,"tp":price + 2.5*risk}
            else:
                if (h_h[rej_idx]>=price) and (h_c[rej_idx]<price) and (uw>=0.5*rng):
                    retest_idx = rej_idx + rej_offset
                    if retest_idx > last: continue
                    if h_h[retest_idx] >= price - tol:
                        sl = h_h[rej_idx] + 4*PIP[code]
                        risk = sl - price
                        return {"pair":code,"dir":"SELL","level":price,
                                "level_type":lv["type"],"rej_ts":h1.index[rej_idx],
                                "signal_ts":h1.index[retest_idx],
                                "entry":price,"sl":sl,"tp":price - 2.5*risk}
    return None


def format_signal(s):
    emoji = "🟢" if s["dir"]=="BUY" else "🔴"
    p = PIP[s["pair"]]
    risk = abs(s["entry"]-s["sl"])/p
    reward = abs(s["tp"]-s["entry"])/p
    return (
        f"{emoji} *MSNR A+ SIGNAL*\n\n"
        f"*Pair:* `{s['pair']}`\n"
        f"*Dir:*  *{s['dir']}*\n"
        f"*Type:* {s['level_type']}\n"
        f"*Time:* `{s['signal_ts'].strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
        f"*Entry:* `{s['entry']:.5f}`\n"
        f"*SL:*    `{s['sl']:.5f}` ({risk:.1f} pips)\n"
        f"*TP:*    `{s['tp']:.5f}` ({reward:.1f} pips)\n"
        f"*R:R:*   1:2.5\n\n"
        f"_MSNR Bot v1.0_"
    )


async def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"🔍 MSNR Scan @ {now} UTC")
    print(f"   Pairs: {', '.join(PAIRS)}\n")

    bot = Bot(token=TELEGRAM_TOKEN)
    found = 0
    for code in PAIRS:
        sig = detect_live_signal(code)
        if sig:
            msg = format_signal(sig)
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, parse_mode="Markdown")
            print(f"   📤 {code} {sig['dir']} @ {sig['level']:.5f}")
            found += 1
        else:
            print(f"   —  {code}: no signal")

    print(f"\n✅ Scan complete. {found} signals found.")


if __name__ == "__main__":
    asyncio.run(main())