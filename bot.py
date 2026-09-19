#!/usr/bin/env python3
"""
MSNR Two-Tier Sniper Bot with State Tracking
"""
import os, json, requests, pandas as pd, numpy as np, asyncio
from datetime import datetime, timezone
from telegram import Bot

TELEGRAM_TOKEN   = os.environ["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
TWELVE_API_KEY   = os.environ["TWELVE_API_KEY"]

PAIRS = ["EURUSD","USDJPY","USDCAD","AUDUSD","NZDUSD","EURJPY"]
PIP = {"EURUSD":0.0001,"USDJPY":0.01,"USDCAD":0.0001,
       "AUDUSD":0.0001,"NZDUSD":0.0001,"EURJPY":0.01}
MAX_SL = {"EURUSD":25,"USDJPY":40,"USDCAD":25,
          "AUDUSD":25,"NZDUSD":25,"EURJPY":40}

A_WICK, A_RETEST, A_BIAS = 0.40, 4, True
B_WICK, B_RETEST, B_BIAS = 0.35, 6, False

SL_BUFFER = 4
RR        = 2.5
STATE_FILE = "state.json"


# ============================================================
# STATE
# ============================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"active": [], "closed": [], "stats": {"total":0,"wins":0,"losses":0}}


def save_state(state):
    state["closed"] = state["closed"][-50:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ============================================================
# TELEGRAM
# ============================================================
async def send_msg(bot, text):
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode="Markdown")
        return True
    except Exception as e:
        print(f"TG ERROR: {e}")
        return False


# ============================================================
# DATA
# ============================================================
def fetch_recent(code, bars=500):
    symbol = f"{code[:3]}/{code[3:]}"
    url = "https://api.twelvedata.com/time_series"
    params = {"symbol":symbol,"interval":"15min","outputsize":bars,
              "order":"ASC","timezone":"UTC","apikey":TWELVE_API_KEY}
    r = requests.get(url, params=params, timeout=20).json()
    if "values" not in r: raise RuntimeError(f"Twelve: {r}")
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c])
    df["volume"] = pd.to_numeric(df.get("volume",0), errors="coerce").fillna(0) if "volume" in df.columns else 0.0
    return df.set_index("datetime").sort_index()


def resample_all(df):
    ohlc = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return {"M15":df,
            "H1": df.resample("1h", label="left", closed="left").agg(ohlc).dropna(),
            "H4": df.resample("4h", label="left", closed="left").agg(ohlc).dropna(),
            "D1": df.resample("1D", label="left", closed="left").agg(ohlc).dropna()}


# ============================================================
# MSNR LOGIC
# ============================================================
def make_levels(tf):
    o=tf["open"].values; c=tf["close"].values; rows=[]
    for i in range(1,len(tf)):
        pb=c[i-1]>o[i-1]; prb=c[i-1]<o[i-1]; cb=c[i]>o[i]; crb=c[i]<o[i]
        if pb and crb: rows.append({"bar":i,"price":c[i-1],"type":"A","dir":"SELL"})
        elif prb and cb: rows.append({"bar":i,"price":c[i-1],"type":"V","dir":"BUY"})
        elif prb and crb: rows.append({"bar":i,"price":c[i-1],"type":"GapS","dir":"SELL"})
        elif pb and cb: rows.append({"bar":i,"price":c[i-1],"type":"GapB","dir":"BUY"})
    return rows


def bias_ok(h4, touch_ts, direction):
    idx = h4.index.searchsorted(touch_ts, side="left")
    if idx < 5: return False
    lb=min(25,idx-1); h=h4["high"].values; l=h4["low"].values; c=h4["close"].values
    if direction=="BUY":
        cause=h[idx-lb:idx].max()
        return bool(np.any(c[max(0,idx-20):min(len(h4),idx+20)]>cause))
    else:
        cause=l[idx-lb:idx].min()
        return bool(np.any(c[max(0,idx-20):min(len(h4),idx+20)]<cause))


def detect_signal(code, h1, h4, anchor_tf, wick, retest_w, use_bias, grade):
    if len(anchor_tf)<20 or len(h1)<10: return None
    levels = make_levels(anchor_tf)
    if not levels: return None
    a_h=anchor_tf["high"].values; a_l=anchor_tf["low"].values
    fresh=[]
    for lv in levels:
        fi=lv["bar"]; p=lv["price"]; d=lv["dir"]; touched=False
        for j in range(fi+1,len(anchor_tf)):
            if d=="SELL" and a_h[j]>=p: touched=True; break
            if d=="BUY" and a_l[j]<=p: touched=True; break
        if not touched: fresh.append(lv)
    if not fresh: return None
    last=len(h1)-2
    if last<3: return None
    h_h=h1["high"].values; h_l=h1["low"].values
    h_o=h1["open"].values; h_c=h1["close"].values
    tol = 5*PIP[code] if PIP[code]==0.01 else 3*PIP[code]
    for rej_offset in range(1, retest_w+1):
        rej_idx = last - (rej_offset-1)
        if rej_idx < 2: continue
        rng = h_h[rej_idx]-h_l[rej_idx]
        if rng<=0: continue
        uw = h_h[rej_idx]-max(h_o[rej_idx],h_c[rej_idx])
        lw = min(h_o[rej_idx],h_c[rej_idx])-h_l[rej_idx]
        for lv in fresh[-15:]:
            price=lv["price"]; d=lv["dir"]
            if use_bias and not bias_ok(h4, h1.index[rej_idx], d): continue
            if d=="BUY":
                if (h_l[rej_idx]<=price) and (h_c[rej_idx]>price) and (lw>=wick*rng):
                    rt=rej_idx+rej_offset
                    if rt>last: continue
                    if h_l[rt]<=price+tol:
                        sl=h_l[rej_idx]-SL_BUFFER*PIP[code]
                        risk=price-sl
                        if risk/PIP[code]>MAX_SL[code] or risk/PIP[code]<5: continue
                        return {"pair":code,"grade":grade,"dir":"BUY","level":price,
                                "level_type":lv["type"],"signal_ts":h1.index[rt],
                                "entry":price,"sl":sl,"tp":price+RR*risk,
                                "risk_pips":risk/PIP[code]}
            else:
                if (h_h[rej_idx]>=price) and (h_c[rej_idx]<price) and (uw>=wick*rng):
                    rt=rej_idx+rej_offset
                    if rt>last: continue
                    if h_h[rt]>=price-tol:
                        sl=h_h[rej_idx]+SL_BUFFER*PIP[code]
                        risk=sl-price
                        if risk/PIP[code]>MAX_SL[code] or risk/PIP[code]<5: continue
                        return {"pair":code,"grade":grade,"dir":"SELL","level":price,
                                "level_type":lv["type"],"signal_ts":h1.index[rt],
                                "entry":price,"sl":sl,"tp":price-RR*risk,
                                "risk_pips":risk/PIP[code]}
    return None


# ============================================================
# MESSAGES
# ============================================================
def fmt_signal(s):
    header = "🎯 *MSNR A+ SNIPER*" if s["grade"]=="A+" else "⚡ *MSNR B+ STANDARD*"
    emoji = "🟢" if s["dir"]=="BUY" else "🔴"
    p = PIP[s["pair"]]
    risk = abs(s["entry"]-s["sl"])/p
    reward = abs(s["tp"]-s["entry"])/p
    tp1 = (s["entry"]+s["tp"])/2
    return (f"{header}\n\n"
            f"{emoji} *{s['pair']}* — *{s['dir']}*\n"
            f"*Type:* {s['level_type']}\n"
            f"*Time:* `{s['signal_ts'].strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
            f"*Entry:* `{s['entry']:.5f}`\n"
            f"*SL:*    `{s['sl']:.5f}`  ({risk:.1f} pips)\n"
            f"*TP1:*   `{tp1:.5f}`  (50%)\n"
            f"*TP:*    `{s['tp']:.5f}`  ({reward:.1f} pips)\n"
            f"*R:R:*   1:{RR}\n\n"
            f"_ID: `{s['id']}`_")


def fmt_tp1(s):
    return (f"🟢 *TP1 HIT — Secure Position*\n\n"
            f"*Pair:* `{s['pair']}`\n"
            f"*Dir:* *{s['dir']}*\n"
            f"*TP1:* `{(s['entry']+s['tp'])/2:.5f}`  (50%)\n\n"
            f"➡️ *Action:* Move SL to Breakeven\n"
            f"➡️ *Action:* Close 50% of position\n\n"
            f"_ID: `{s['id']}`_")


def fmt_tp(s):
    p = PIP[s["pair"]]
    profit = abs(s["tp"]-s["entry"])/p
    return (f"🎉 *TP HIT — WIN*\n\n"
            f"*Pair:* `{s['pair']}`\n"
            f"*Dir:* *{s['dir']}*\n"
            f"*Result:* *+{profit:.1f} pips*\n"
            f"*Close Time:* `{s['close_ts'][:19]}`\n\n"
            f"_ID: `{s['id']}`_")


def fmt_sl(s):
    p = PIP[s["pair"]]
    loss = abs(s["entry"]-s["sl"])/p
    return (f"❌ *SL HIT — LOSS*\n\n"
            f"*Pair:* `{s['pair']}`\n"
            f"*Dir:* *{s['dir']}*\n"
            f"*Result:* *-{loss:.1f} pips*\n"
            f"*Close Time:* `{s['close_ts'][:19]}`\n\n"
            f"_ID: `{s['id']}`_")


# ============================================================
# TRACK ACTIVE SIGNALS
# ============================================================
async def track_active(bot, state, m15_dict):
    still_active = []
    for sig in state["active"]:
        code = sig["pair"]
        if code not in m15_dict:
            still_active.append(sig); continue
        m15 = m15_dict[code]
        signal_ts = pd.to_datetime(sig["signal_ts"], utc=True)
        after = m15[m15.index > signal_ts]
        if len(after)==0:
            still_active.append(sig); continue

        entry=sig["entry"]; sl=sig["sl"]; tp=sig["tp"]
        tp1=(entry+tp)/2
        d=sig["dir"]; closed=False

        for ts, bar in after.iterrows():
            hi=bar["high"]; lo=bar["low"]
            if d=="BUY":
                if not sig["tp1_hit"] and lo<=sl:
                    sig["status"]="closed_sl"; sig["close_ts"]=ts.isoformat()
                    await send_msg(bot, fmt_sl(sig))
                    state["stats"]["losses"]+=1; closed=True; break
                if not sig["tp1_hit"] and hi>=tp1:
                    sig["tp1_hit"]=True; sig["tp1_ts"]=ts.isoformat()
                    await send_msg(bot, fmt_tp1(sig))
                if hi>=tp:
                    sig["status"]="closed_tp"; sig["close_ts"]=ts.isoformat()
                    await send_msg(bot, fmt_tp(sig))
                    state["stats"]["wins"]+=1; closed=True; break
                if sig["tp1_hit"] and lo<=entry:
                    sig["status"]="closed_be"; sig["close_ts"]=ts.isoformat()
                    closed=True; break
            else:  # SELL
                if not sig["tp1_hit"] and hi>=sl:
                    sig["status"]="closed_sl"; sig["close_ts"]=ts.isoformat()
                    await send_msg(bot, fmt_sl(sig))
                    state["stats"]["losses"]+=1; closed=True; break
                if not sig["tp1_hit"] and lo<=tp1:
                    sig["tp1_hit"]=True; sig["tp1_ts"]=ts.isoformat()
                    await send_msg(bot, fmt_tp1(sig))
                if lo<=tp:
                    sig["status"]="closed_tp"; sig["close_ts"]=ts.isoformat()
                    await send_msg(bot, fmt_tp(sig))
                    state["stats"]["wins"]+=1; closed=True; break
                if sig["tp1_hit"] and hi>=entry:
                    sig["status"]="closed_be"; sig["close_ts"]=ts.isoformat()
                    closed=True; break

        if closed:
            state["closed"].append(sig)
        else:
            still_active.append(sig)

    state["active"] = still_active


# ============================================================
# MAIN
# ============================================================
async def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"MSNR Scan @ {now}")
    bot = Bot(token=TELEGRAM_TOKEN)
    state = load_state()

    # Fetch data
    m15_dict = {}
    for code in PAIRS:
        try:
            m15_dict[code] = fetch_recent(code, bars=500)
        except Exception as e:
            print(f"WARN {code}: {e}")

    # 1) Track existing signals first
    await track_active(bot, state, m15_dict)

    # 2) Detect new signals
    active_ids = {s["id"] for s in state["active"]}
    for code in PAIRS:
        if code not in m15_dict: continue
        tf = resample_all(m15_dict[code])
        h1, h4, d1 = tf["H1"], tf["H4"], tf["D1"]
        sig = detect_signal(code, h1, h4, d1, A_WICK, A_RETEST, A_BIAS, "A+")
        if sig is None:
            sig = detect_signal(code, h1, h4, h4, B_WICK, B_RETEST, B_BIAS, "B+")
        if sig:
            sig["id"] = f"{code}_{sig['signal_ts'].strftime('%Y%m%d_%H%M')}"
            if sig["id"] in active_ids: continue
            sig["tp1_hit"]=False; sig["status"]="active"
            sig["signal_ts"]=sig["signal_ts"].isoformat()
            state["active"].append(sig)
            state["stats"]["total"]+=1
            await send_msg(bot, fmt_signal({
                **sig,
                "signal_ts": pd.to_datetime(sig["signal_ts"])
            }))
            print(f"NEW SIGNAL: {sig['id']}")
        else:
            print(f"-- {code}: no signal")

    save_state(state)
    print(f"Done. Active: {len(state['active'])} | Closed: {len(state['closed'])}")


if __name__ == "__main__":
    asyncio.run(main())