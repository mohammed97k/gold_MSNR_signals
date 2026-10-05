import os
import json
import requests
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

# ==================== الإعدادات (مطابقة للـ Workflow) ====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TELEGRAM_GROUP_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_API_KEY")

STATE_FILE = "state.json"
NY_TZ = ZoneInfo("America/New_York")
MOSUL_TZ = ZoneInfo("Asia/Baghdad")
SYMBOL = "XAU/USD"

# ==================== الثوابت ====================
MULT = 10.0
ATR_PERIOD = 14
MIN_SL_PTS = 40.0
MAX_SL_PTS = 150.0
COOLDOWN_NORMAL = 10
COOLDOWN_STRONG = 3
MAX_TRADES_PER_DAY = 3
MAX_BARS_TRADE = 60
TP1_R = 1.0
TP2_R = 2.0
TP3_R = 3.0
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
SWEEP_LB = 20
FVG_EXPIRY = 15
OB_EXPIRY = 15
MSS_EXPIRY = 15
SWEEP_EXPIRY = 20
BRK_EXPIRY = 20
STALE_DATA_MINUTES = 45


def fmt_mosul(dt_utc):
    return dt_utc.astimezone(MOSUL_TZ).strftime('%I:%M %p')


def send_telegram(message):
    chat_ids = [c for c in [TELEGRAM_CHAT_ID, TELEGRAM_GROUP_CHAT_ID] if c]
    for chat_id in chat_ids:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
            print(f"TG->{chat_id}: ok={r.json().get('ok')}")
        except Exception as e:
            print(f"TG Error: {e}")


# ==================== جلب البيانات من Twelve Data ====================
def fetch_candles(symbol, interval, outputsize=500):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "format": "JSON",
        "timezone": "UTC"
    }
    try:
        r = requests.get(url, params=params, timeout=60)
        if r.status_code != 200:
            print(f"❌ فشل جلب {interval}: {r.status_code} {r.text[:200]}")
            return None
        data = r.json()
        if "values" not in data:
            print(f"❌ Twelve Data error: {data.get('message', 'unknown')}")
            return None
        candles = data["values"]
        df = pd.DataFrame(candles)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df[["datetime", "open", "high", "low", "close"]].copy()
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c])
        df = df.sort_values("datetime").reset_index(drop=True)
        print(f"✅ {interval}: {len(df)} شمعة | آخر: {df.iloc[-1]['close']}")
        return df
    except Exception as e:
        print(f"❌ Fetch {interval} Error: {e}")
        traceback.print_exc()
        return None


# ==================== المؤشرات ====================
def ta_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0/period, adjust=False).mean()


def ta_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def ta_pivothigh(highs, left, right):
    n = len(highs)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        window = highs[i-left:i+right+1]
        if highs[i] == window.max() and (window == highs[i]).sum() == 1:
            result[i + right] = highs[i]
    return result


def ta_pivotlow(lows, left, right):
    n = len(lows)
    result = np.full(n, np.nan)
    for i in range(left, n - right):
        window = lows[i-left:i+right+1]
        if lows[i] == window.min() and (window == lows[i]).sum() == 1:
            result[i + right] = lows[i]
    return result


def ta_lowest(series, length):
    return series.rolling(length).min()


def ta_highest(series, length):
    return series.rolling(length).max()


# ==================== الجلسات ====================
def tm(dt, h1, m1, h2, m2):
    t = dt.hour * 60 + dt.minute
    return (h1 * 60 + m1) <= t < (h2 * 60 + m2)


def session_flags(dt):
    return {
        "LondonO": tm(dt, 2, 0, 3, 0),
        "SB-LDN":  tm(dt, 3, 0, 4, 0),
        "Judas":   tm(dt, 9, 30, 10, 0),
        "SB-AM":   tm(dt, 10, 0, 11, 0),
        "2022-AM": tm(dt, 11, 0, 11, 30),
        "Lunch":   tm(dt, 11, 50, 12, 10),
        "SB-PM":   tm(dt, 14, 0, 15, 0),
        "MOC":     tm(dt, 15, 15, 15, 45),
        "FOMC":    tm(dt, 14, 0, 14, 30),
        "NFP":     tm(dt, 8, 30, 9, 0),
        "TGIF":    (dt.weekday() == 4) and tm(dt, 14, 0, 15, 0),
    }


def is_blackout(dt):
    return tm(dt, 12, 10, 13, 59)


# ==================== الحالة ====================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            s = json.load(f)
            print(f"📖 State: active_trade={'✅' if s.get('active_trade') else '❌'} | trades_today={s.get('trade_count_today', 0)}")
            return s
    print("📖 State file جديد")
    return {"active_trade": None, "last_entry_time": None, "trade_count_today": 0, "last_day": None, "models_done_today": []}


def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)
    print(f"💾 State saved: active_trade={'✅' if s.get('active_trade') else '❌'}")


# ==================== إدارة الصفقة ====================
def manage_trade(state, current_price, now_utc):
    print(f"   [manage_trade] بدء...")
    t = state["active_trade"]
    if t is None:
        print(f"   [manage_trade] لا صفقة نشطة")
        return

    d, e, sl, tp1, tp2, tp3 = t["direction"], t["entry"], t["sl"], t["tp1"], t["tp2"], t["tp3"]
    time_str = fmt_mosul(now_utc)
    print(f"   [manage_trade] {t.get('model')} {d} @ {e} | SL={sl} TP1={tp1} TP2={tp2} TP3={tp3}")

    entry_time_str = t.get("entry_time")
    if entry_time_str:
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            elapsed = (now_utc - entry_dt).total_seconds()
            bars_elapsed = elapsed / 300
            if bars_elapsed >= MAX_BARS_TRADE:
                pnl_pts = (e - current_price) if d == "SELL" else (current_price - e)
                pnl_pts *= MULT
                send_telegram(f"⏰ خروج بالوقت (60 شمعة)\n{t['model']} {d}\n{e}\nPnL: {pnl_pts:+.2f} pts\n{time_str}")
                state["active_trade"] = None
                return
        except Exception as e2:
            print(f"   [manage_trade] ⚠️ Time error: {e2}")

    if (d == "BUY" and current_price <= sl) or (d == "SELL" and current_price >= sl):
        if t["tp1_hit"]:
            send_telegram(f"⚖️ ضرب الستوب بعد TP1 (BE)\n{t['model']} {d}\n{e} | {time_str}")
        else:
            send_telegram(f"🛑 ضرب الستوب!\n{t['model']} {d}\n{e}\n{time_str}")
        state["active_trade"] = None
        return

    if not t["tp1_hit"] and ((d == "BUY" and current_price >= tp1) or (d == "SELL" and current_price <= tp1)):
        t["tp1_hit"] = True
        send_telegram(f"🎯 الهدف الأول!\n{t['model']} {d}\nTP1: {tp1}\n{time_str}")

    if t["tp1_hit"] and not t["tp2_hit"] and ((d == "BUY" and current_price >= tp2) or (d == "SELL" and current_price <= tp2)):
        t["tp2_hit"] = True
        send_telegram(f"🎯🎯 الهدف الثاني!\n{t['model']}\nTP2: {tp2}\n{time_str}")

    if t["tp2_hit"] and not t["tp3_hit"] and ((d == "BUY" and current_price >= tp3) or (d == "SELL" and current_price <= tp3)):
        t["tp3_hit"] = True
        send_telegram(f"🎯🎯🎯 الهدف الثالث! دبچة 🕺\n{t['model']}\nTP3: {tp3}\n{time_str}")
        state["active_trade"] = None


# ==================== بناء السياق ====================
def build_context(df):
    n = len(df)
    atr = ta_atr(df, ATR_PERIOD).values
    sh = ta_pivothigh(df["high"].values, PIVOT_LEFT, PIVOT_RIGHT)
    sl = ta_pivotlow(df["low"].values, PIVOT_LEFT, PIVOT_RIGHT)

    last_sh = np.full(n, np.nan)
    last_sl = np.full(n, np.nan)
    cur_sh, cur_sl = np.nan, np.nan
    for i in range(n):
        if not np.isnan(sh[i]): cur_sh = sh[i]
        if not np.isnan(sl[i]): cur_sl = sl[i]
        last_sh[i] = cur_sh
        last_sl[i] = cur_sl

    prev_sh = np.full(n, np.nan)
    prev_sl = np.full(n, np.nan)
    psh, psl = np.nan, np.nan
    sh_seen, sl_seen = np.nan, np.nan
    for i in range(n):
        if not np.isnan(sh[i]):
            if not np.isnan(sh_seen): psh = sh_seen
            sh_seen = sh[i]
        if not np.isnan(sl[i]):
            if not np.isnan(sl_seen): psl = sl_seen
            sl_seen = sl[i]
        prev_sh[i] = psh
        prev_sl[i] = psl

    recent_low = ta_lowest(df["low"], SWEEP_LB).shift(1).values
    recent_high = ta_highest(df["high"], SWEEP_LB).shift(1).values
    bull_sweep = (df["low"].values < recent_low) & (df["close"].values > recent_low)
    bear_sweep = (df["high"].values > recent_high) & (df["close"].values < recent_high)

    bull_sweep_ok = np.zeros(n, dtype=bool); bull_sweep_low = np.full(n, np.nan)
    bear_sweep_ok = np.zeros(n, dtype=bool); bear_sweep_high = np.full(n, np.nan)
    cbok, cbbr, cbbl = False, -1, np.nan
    cek, cebr, cebh = False, -1, np.nan
    for i in range(n):
        if bull_sweep[i]: cbok, cbbr, cbbl = True, i, df["low"].iloc[i]
        if bear_sweep[i]: cek, cebr, cebh = True, i, df["high"].iloc[i]
        if cbok and (i - cbbr) > SWEEP_EXPIRY: cbok = False
        if cek and (i - cebr) > SWEEP_EXPIRY: cek = False
        bull_sweep_ok[i], bull_sweep_low[i] = cbok, cbbl
        bear_sweep_ok[i], bear_sweep_high[i] = cek, cebh

    tick = 0.01
    shallowBull = (df["low"].values < recent_low) & (df["low"].values > (recent_low - 3 * tick)) & (df["close"].values > recent_low)
    shallowBear = (df["high"].values > recent_high) & (df["high"].values < (recent_high + 3 * tick)) & (df["close"].values < recent_high)

    eq_tol = atr * 0.1
    eqHighs = np.zeros(n, dtype=bool); eqLows = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(prev_sh[i]) and not np.isnan(last_sh[i]):
            if abs(prev_sh[i] - last_sh[i]) < eq_tol[i]: eqHighs[i] = True
        if not np.isnan(prev_sl[i]) and not np.isnan(last_sl[i]):
            if abs(prev_sl[i] - last_sl[i]) < eq_tol[i]: eqLows[i] = True

    bull_mss = np.zeros(n, dtype=bool); bull_mss_bar = np.full(n, -1, dtype=int)
    bear_mss = np.zeros(n, dtype=bool); bear_mss_bar = np.full(n, -1, dtype=int)
    cbm, cbmbar = False, -1; csm, csmbar = False, -1
    close_a = df["close"].values
    for i in range(1, n):
        if not np.isnan(last_sh[i]) and close_a[i] > last_sh[i] and close_a[i-1] <= last_sh[i] and bull_sweep_ok[i]:
            cbm, cbmbar = True, i
        if not np.isnan(last_sl[i]) and close_a[i] < last_sl[i] and close_a[i-1] >= last_sl[i] and bear_sweep_ok[i]:
            csm, csmbar = True, i
        if cbm and (i - cbmbar) > MSS_EXPIRY: cbm = False
        if csm and (i - csmbar) > MSS_EXPIRY: csm = False
        bull_mss[i], bull_mss_bar[i] = cbm, cbmbar
        bear_mss[i], bear_mss_bar[i] = csm, csmbar

    bTop = np.full(n, np.nan); bBot = np.full(n, np.nan); bActive = np.zeros(n, dtype=bool)
    sTop = np.full(n, np.nan); sBot = np.full(n, np.nan); sActive = np.zeros(n, dtype=bool)
    cbt, cbb, cbbar, cbact = np.nan, np.nan, -1, False
    cst, csb, csbar, csact = np.nan, np.nan, -1, False
    low_a = df["low"].values; high_a = df["high"].values
    for i in range(2, n):
        if low_a[i] > high_a[i-2]:
            cbt, cbb, cbbar, cbact = low_a[i], high_a[i-2], i, True
        if high_a[i] < low_a[i-2]:
            cst, csb, csbar, csact = low_a[i-2], high_a[i], i, True
        if cbact and (i - cbbar) > FVG_EXPIRY: cbact = False
        if csact and (i - csbar) > FVG_EXPIRY: csact = False
        bTop[i], bBot[i], bActive[i] = cbt, cbb, cbact
        sTop[i], sBot[i], sActive[i] = cst, csb, csact
    bCE = np.where(bActive, (bTop + bBot) / 2.0, np.nan)
    sCE = np.where(sActive, (sTop + sBot) / 2.0, np.nan)

    bOBHigh = np.full(n, np.nan); bOBLow = np.full(n, np.nan); bOBMT = np.full(n, np.nan)
    bOBActive = np.zeros(n, dtype=bool)
    sOBHigh = np.full(n, np.nan); sOBLow = np.full(n, np.nan); sOBMT = np.full(n, np.nan)
    sOBActive = np.zeros(n, dtype=bool)
    o = df["open"].values; h = df["high"].values; l = df["low"].values; c = df["close"].values
    cboH, cboL, cboMT, cboBar, cboAct = np.nan, np.nan, np.nan, -1, False
    csoH, csoL, csoMT, csoBar, csoAct = np.nan, np.nan, np.nan, -1, False
    for i in range(1, n):
        if c[i] > o[i] and c[i-1] < o[i-1]:
            cboH, cboL = h[i-1], l[i-1]; cboMT = (o[i-1] + c[i-1]) / 2.0; cboBar, cboAct = i-1, True
        if c[i] < o[i] and c[i-1] > o[i-1]:
            csoH, csoL = h[i-1], l[i-1]; csoMT = (o[i-1] + c[i-1]) / 2.0; csoBar, csoAct = i-1, True
        if cboAct and (i - cboBar) > OB_EXPIRY: cboAct = False
        if csoAct and (i - csoBar) > OB_EXPIRY: csoAct = False
        bOBHigh[i], bOBLow[i], bOBMT[i], bOBActive[i] = cboH, cboL, cboMT, cboAct
        sOBHigh[i], sOBLow[i], sOBMT[i], sOBActive[i] = csoH, csoL, csoMT, csoAct

    bBrkHigh = np.full(n, np.nan); bBrkLow = np.full(n, np.nan); bBrkActive = np.zeros(n, dtype=bool)
    sBrkHigh = np.full(n, np.nan); sBrkLow = np.full(n, np.nan); sBrkActive = np.zeros(n, dtype=bool)
    cbbrh, cbbrl, cbbrbar, cbbract = np.nan, np.nan, -1, False
    csbrh, csbrl, csbrbar, csbract = np.nan, np.nan, -1, False
    for i in range(n):
        if bull_mss[i] and bull_sweep_ok[i]:
            cbbrh, cbbrl = last_sh[i], bull_sweep_low[i]
            cbbrbar, cbbract = i, True
        if bear_mss[i] and bear_sweep_ok[i]:
            csbrh, csbrl = bear_sweep_high[i], last_sl[i]
            csbrbar, csbract = i, True
        if cbbract and (i - cbbrbar) > BRK_EXPIRY: cbbract = False
        if csbract and (i - csbrbar) > BRK_EXPIRY: csbract = False
        bBrkHigh[i], bBrkLow[i], bBrkActive[i] = cbbrh, cbbrl, cbbract
        sBrkHigh[i], sBrkLow[i], sBrkActive[i] = csbrh, csbrl, csbract
    bBrkCE = np.where(bBrkActive, (bBrkHigh + bBrkLow) / 2.0, np.nan)
    sBrkCE = np.where(sBrkActive, (sBrkHigh + sBrkLow) / 2.0, np.nan)

    bProp = bOBActive & (c < o) & (h <= bOBHigh) & (l >= bOBLow)
    sProp = sOBActive & (c > o) & (l >= sOBLow) & (h <= sOBHigh)

    high_s2 = np.concatenate([np.full(2, np.nan), high_a[:-2]])
    low_s2 = np.concatenate([np.full(2, np.nan), low_a[:-2]])
    with np.errstate(invalid='ignore'):
        lqVoidUp = (low_a > high_s2) & ((low_a - high_s2) >= atr * 1.5)
        lqVoidDn = (high_a < low_s2) & ((low_s2 - high_a) >= atr * 1.5)
    lqVoidUp = np.nan_to_num(lqVoidUp).astype(bool)
    lqVoidDn = np.nan_to_num(lqVoidDn).astype(bool)

    floatUp = (ta_highest(df["high"], 20).values > ta_highest(df["high"], 40).shift(10).values)
    floatDn = (ta_lowest(df["low"], 20).values < ta_lowest(df["low"], 40).shift(10).values)

    bprBull = bActive & sActive & (bBot <= sTop) & (bTop >= sBot)
    bprBear = bActive & sActive & (sBot <= bTop) & (sTop >= bBot)

    rejTouchBull = np.zeros(n, dtype=bool); rejTouchBear = np.zeros(n, dtype=bool)
    for i in range(2, n):
        rejBlock = (h[i-1] < h[i-2]) and (c[i-1] > o[i-1])
        if rejBlock and l[i] <= l[i-1] and c[i] > l[i-1]: rejTouchBull[i] = True
        rejBlockS = (l[i-1] > l[i-2]) and (c[i-1] < o[i-1])
        if rejBlockS and h[i] >= h[i-1] and c[i] < h[i-1]: rejTouchBear[i] = True

    mitBull = bBrkActive & (c < o) & (l <= bBrkCE) & (l >= bBrkLow)
    mitBear = sBrkActive & (c > o) & (h >= sBrkCE) & (h <= sBrkHigh)

    df_copy = df.copy()
    df_copy["date_ny"] = df_copy["datetime"].dt.tz_convert(NY_TZ).dt.date
    df_copy["high_d"] = df_copy.groupby("date_ny")["high"].transform("max")
    df_copy["low_d"] = df_copy.groupby("date_ny")["low"].transform("min")
    daily_high = df_copy.groupby("date_ny")["high_d"].last().shift(1)
    daily_low = df_copy.groupby("date_ny")["low_d"].last().shift(1)
    df_copy["asia_hi"] = df_copy["date_ny"].map(daily_high)
    df_copy["asia_lo"] = df_copy["date_ny"].map(daily_low)
    asia_hi_arr = df_copy["asia_hi"].values
    asia_lo_arr = df_copy["asia_lo"].values
    with np.errstate(invalid='ignore'):
        p3Bull = (low_a < asia_lo_arr) & (c > asia_lo_arr) & bull_mss
        p3Bear = (high_a > asia_hi_arr) & (c < asia_hi_arr) & bear_mss
    p3Bull = np.nan_to_num(p3Bull).astype(bool)
    p3Bear = np.nan_to_num(p3Bear).astype(bool)

    q4 = (df["datetime"].dt.month >= 10).values

    ndogCE = np.full(n, np.nan); orgCE = np.full(n, np.nan)
    df_ny = df["datetime"].dt.tz_convert(NY_TZ)
    for i in range(n):
        dt_ny = df_ny.iloc[i]
        if dt_ny.hour == 18 and dt_ny.minute == 0 and i > 0:
            ndogCE[i] = (c[i] + c[i-1]) / 2.0
        elif i > 0 and not np.isnan(ndogCE[i-1]):
            ndogCE[i] = ndogCE[i-1]
        if dt_ny.hour == 9 and dt_ny.minute == 30 and i > 0:
            orgCE[i] = (c[i] + c[i-1]) / 2.0
        elif i > 0 and not np.isnan(orgCE[i-1]):
            orgCE[i] = orgCE[i-1]

    with np.errstate(invalid='ignore'):
        ndogTouchBull = (~np.isnan(ndogCE)) & (low_a <= ndogCE) & (c > ndogCE)
        ndogTouchBear = (~np.isnan(ndogCE)) & (high_a >= ndogCE) & (c < ndogCE)
        orgTouchBull = (~np.isnan(orgCE)) & (low_a <= orgCE) & (c > orgCE)
        orgTouchBear = (~np.isnan(orgCE)) & (high_a >= orgCE) & (c < orgCE)
    ndogTouchBull = np.nan_to_num(ndogTouchBull).astype(bool); ndogTouchBear = np.nan_to_num(ndogTouchBear).astype(bool)
    orgTouchBull = np.nan_to_num(orgTouchBull).astype(bool); orgTouchBear = np.nan_to_num(orgTouchBear).astype(bool)

    recentBullMSS = bull_mss & ((np.arange(n) - bull_mss_bar) <= 10)
    recentBearMSS = bear_mss & ((np.arange(n) - bear_mss_bar) <= 10)

    return {
        "atr": atr, "last_sh": last_sh, "last_sl": last_sl,
        "prev_sh": prev_sh, "prev_sl": prev_sl,
        "bull_sweep_ok": bull_sweep_ok, "bear_sweep_ok": bear_sweep_ok,
        "bull_sweep_low": bull_sweep_low, "bear_sweep_high": bear_sweep_high,
        "bull_mss": bull_mss, "bear_mss": bear_mss,
        "bTop": bTop, "bBot": bBot, "bActive": bActive,
        "sTop": sTop, "sBot": sBot, "sActive": sActive,
        "bCE": bCE, "sCE": sCE,
        "bOBHigh": bOBHigh, "bOBLow": bOBLow, "bOBMT": bOBMT, "bOBActive": bOBActive,
        "sOBHigh": sOBHigh, "sOBLow": sOBLow, "sOBMT": sOBMT, "sOBActive": sOBActive,
        "bBrkActive": bBrkActive, "sBrkActive": sBrkActive,
        "bBrkCE": bBrkCE, "sBrkCE": sBrkCE,
        "bBrkLow": bBrkLow, "sBrkHigh": sBrkHigh,
        "eqHighs": eqHighs, "eqLows": eqLows,
        "shallowBull": shallowBull, "shallowBear": shallowBear,
        "floatUp": floatUp, "floatDn": floatDn,
        "bprBull": bprBull, "bprBear": bprBear,
        "rejTouchBull": rejTouchBull, "rejTouchBear": rejTouchBear,
        "lqVoidUp": lqVoidUp, "lqVoidDn": lqVoidDn,
        "bProp": bProp, "sProp": sProp,
        "mitBull": mitBull, "mitBear": mitBear,
        "p3Bull": p3Bull, "p3Bear": p3Bear,
        "q4": q4,
        "ndogTouchBull": ndogTouchBull, "ndogTouchBear": ndogTouchBear,
        "orgTouchBull": orgTouchBull, "orgTouchBear": orgTouchBear,
        "recentBullMSS": recentBullMSS, "recentBearMSS": recentBearMSS,
    }


# ==================== فحص الإشارة ====================
def check_signal(df5, df1h, state, now_utc, now_ny):
    print("   [check_signal] بدء...")
    try:
        ctx = build_context(df5)
        i = len(df5) - 1
        atr = ctx["atr"][i]
        if np.isnan(atr):
            return None

        df1h = df1h.copy()
        df1h["ema200"] = ta_ema(df1h["close"], 200)
        df1h["ema50"] = ta_ema(df1h["close"], 50)
        h1 = df1h.iloc[-2]
        trend_up = h1["close"] > h1["ema200"] if not pd.isna(h1["ema200"]) else False
        trend_down = h1["close"] < h1["ema200"] if not pd.isna(h1["ema200"]) else False
        strong_bull = trend_up and (not pd.isna(h1["ema50"])) and h1["close"] > h1["ema50"]
        strong_bear = trend_down and (not pd.isna(h1["ema50"])) and h1["close"] < h1["ema50"]
        print(f"   [check_signal] H1: Trend={'UP' if trend_up else 'DOWN' if trend_down else 'NONE'}")

        sessions = session_flags(now_ny)
        any_kz = any([
            sessions["LondonO"], sessions["SB-LDN"], sessions["Judas"],
            sessions["SB-AM"], sessions["2022-AM"], sessions["Lunch"],
            sessions["SB-PM"], sessions["MOC"], sessions["FOMC"],
            sessions["NFP"], sessions["TGIF"]
        ])
        sessOK = any_kz and not is_blackout(now_ny)

        if not sessOK:
            print(f"   [check_signal] ❌ لا جلسة نشطة")
            return None

        if state["trade_count_today"] >= MAX_TRADES_PER_DAY:
            return None

        if state["last_entry_time"]:
            last_dt = datetime.fromisoformat(state["last_entry_time"])
            bars_since = int((now_utc - last_dt).total_seconds() / 300)
            cd = COOLDOWN_STRONG if (strong_bull or strong_bear) else COOLDOWN_NORMAL
            if bars_since <= cd:
                return None

        canEnter = True
        trendL = trend_up
        trendS = trend_down

        s1L = canEnter and sessions["SB-LDN"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s1S = canEnter and sessions["SB-LDN"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s2L = canEnter and sessions["Judas"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s2S = canEnter and sessions["Judas"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s3L = canEnter and sessions["SB-AM"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s3S = canEnter and sessions["SB-AM"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s4L = canEnter and sessions["2022-AM"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s4S = canEnter and sessions["2022-AM"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s5L = canEnter and sessions["Lunch"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s5S = canEnter and sessions["Lunch"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s6L = canEnter and sessions["SB-PM"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s6S = canEnter and sessions["SB-PM"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s7L = canEnter and sessions["MOC"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s7S = canEnter and sessions["MOC"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s10L = canEnter and sessions["Judas"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s10S = canEnter and sessions["Judas"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS
        s11L = canEnter and sessions["FOMC"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s11S = canEnter and sessions["FOMC"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s17L = canEnter and ctx["bOBActive"][i] and ctx["bOBLow"][i] <= df5["low"].iloc[i] <= ctx["bOBMT"][i] and trendL
        s17S = canEnter and ctx["sOBActive"][i] and ctx["sOBMT"][i] <= df5["high"].iloc[i] <= ctx["sOBHigh"][i] and trendS

        s18L = canEnter and ctx["bProp"][i] and ctx["bull_mss"][i] and trendL
        s18S = canEnter and ctx["sProp"][i] and ctx["bear_mss"][i] and trendS

        s20L = canEnter and ctx["mitBull"][i] and trendL
        s20S = canEnter and ctx["mitBear"][i] and trendS

        s22L = canEnter and ctx["eqLows"][i] and ctx["bull_sweep_ok"][i] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s22S = canEnter and ctx["eqHighs"][i] and ctx["bear_sweep_ok"][i] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s24L = canEnter and ctx["shallowBull"][i] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s24S = canEnter and ctx["shallowBear"][i] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s26L = canEnter and ctx["floatUp"][i] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s26S = canEnter and ctx["floatDn"][i] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s27L = canEnter and ctx["ndogTouchBull"][i] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s27S = canEnter and ctx["ndogTouchBear"][i] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s29L = canEnter and ctx["orgTouchBull"][i] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s29S = canEnter and ctx["orgTouchBear"][i] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s30L = canEnter and ctx["bprBull"][i] and ctx["bull_mss"][i] and trendL
        s30S = canEnter and ctx["bprBear"][i] and ctx["bear_mss"][i] and trendS

        s32L = canEnter and ctx["rejTouchBull"][i] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s32S = canEnter and ctx["rejTouchBear"][i] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s33L = canEnter and ctx["lqVoidUp"][i] and ctx["bull_mss"][i] and trendL
        s33S = canEnter and ctx["lqVoidDn"][i] and ctx["bear_mss"][i] and trendS

        s35L = canEnter and (now_ny.weekday() == 4) and sessions["TGIF"] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s35S = canEnter and (now_ny.weekday() == 4) and sessions["TGIF"] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s36L = canEnter and ctx["q4"][i] and ctx["bull_mss"][i] and ctx["bActive"][i] and trendL
        s36S = canEnter and ctx["q4"][i] and ctx["bear_mss"][i] and ctx["sActive"][i] and trendS

        s37L = canEnter and ctx["p3Bull"][i] and ctx["bActive"][i] and trendL
        s37S = canEnter and ctx["p3Bear"][i] and ctx["sActive"][i] and trendS

        tcpL = canEnter and strong_bull and ctx["recentBullMSS"][i] and ctx["bActive"][i] and df5["low"].iloc[i] <= ctx["bCE"][i] and df5["close"].iloc[i] > ctx["bCE"][i] and df5["low"].iloc[i] > ctx["bBot"][i]
        tcpS = canEnter and strong_bear and ctx["recentBearMSS"][i] and ctx["sActive"][i] and df5["high"].iloc[i] >= ctx["sCE"][i] and df5["close"].iloc[i] < ctx["sCE"][i] and df5["high"].iloc[i] < ctx["sTop"][i]

        fvgL = canEnter and strong_bull and ctx["bActive"][i] and df5["low"].iloc[i] <= ctx["bCE"][i] and df5["close"].iloc[i] > ctx["bCE"][i] and df5["close"].iloc[i] > ctx["bBot"][i]
        fvgS = canEnter and strong_bear and ctx["sActive"][i] and df5["high"].iloc[i] >= ctx["sCE"][i] and df5["close"].iloc[i] < ctx["sCE"][i] and df5["close"].iloc[i] < ctx["sTop"][i]

        recentLowBreak = strong_bear and df5["low"].iloc[i] < ta_lowest(df5["low"], 10).shift(1).iloc[i]
        recentHighBreak = strong_bull and df5["high"].iloc[i] > ta_highest(df5["high"], 10).shift(1).iloc[i]
        bosL = canEnter and strong_bull and recentHighBreak and df5["close"].iloc[i] < df5["close"].iloc[i-1] and ctx["bActive"][i] and df5["low"].iloc[i] <= ctx["bCE"][i] and df5["close"].iloc[i] > ctx["bCE"][i]
        bosS = canEnter and strong_bear and recentLowBreak and df5["close"].iloc[i] > df5["close"].iloc[i-1] and ctx["sActive"][i] and df5["high"].iloc[i] >= ctx["sCE"][i] and df5["close"].iloc[i] < ctx["sCE"][i]

        sigL = s1L or s2L or s3L or s4L or s5L or s6L or s7L or s10L or s11L or s17L or s18L or s20L or s22L or s24L or s26L or s27L or s29L or s30L or s32L or s33L or s35L or s36L or s37L or tcpL or fvgL or bosL
        sigS = s1S or s2S or s3S or s4S or s5S or s6S or s7S or s10S or s11S or s17S or s18S or s20S or s22S or s24S or s26S or s27S or s29S or s30S or s32S or s33S or s35S or s36S or s37S or tcpS or fvgS or bosS
        print(f"   [check_signal] sigL={sigL} sigS={sigS}")

        if not (sigL or sigS):
            return None

        model = "Unknown"
        if s1L or s1S: model = "SB-LDN"
        elif s2L or s2S: model = "Judas"
        elif s3L or s3S: model = "SB-AM"
        elif s4L or s4S: model = "2022-AM"
        elif s5L or s5S: model = "Lunch"
        elif s6L or s6S: model = "SB-PM"
        elif s7L or s7S: model = "MOC"
        elif s10L or s10S: model = "NY-Open"
        elif s11L or s11S: model = "FOMC"
        elif s17L or s17S: model = "OB"
        elif s18L or s18S: model = "Prop"
        elif s20L or s20S: model = "Mitig"
        elif s22L or s22S: model = "EQL"
        elif s24L or s24S: model = "Shallow"
        elif s26L or s26S: model = "Float"
        elif s27L or s27S: model = "NDOG"
        elif s29L or s29S: model = "ORG"
        elif s30L or s30S: model = "BPR"
        elif s32L or s32S: model = "Reject"
        elif s33L or s33S: model = "Void"
        elif s35L or s35S: model = "TGIF"
        elif s36L or s36S: model = "Quarter"
        elif s37L or s37S: model = "P3"
        elif tcpL or tcpS: model = "TCP"
        elif fvgL or fvgS: model = "FVG"
        elif bosL or bosS: model = "BOS"

        close = df5["close"].iloc[i]
        if sigL:
            if (s17L or s18L) and ctx["bOBActive"][i]:
                entry = ctx["bOBMT"][i]; sl_level = ctx["bOBLow"][i] - atr * 0.3
            elif (fvgL or bosL) and ctx["bActive"][i]:
                entry = ctx["bCE"][i]; sl_level = ctx["bBot"][i] - atr * 0.3
            elif ctx["bActive"][i]:
                entry = ctx["bCE"][i]; sl_level = ctx["bull_sweep_low"][i] - atr * 0.3
            else:
                entry = close; sl_level = ctx["bull_sweep_low"][i] - atr * 0.3
            direction = "BUY"
        else:
            if (s17S or s18S) and ctx["sOBActive"][i]:
                entry = ctx["sOBMT"][i]; sl_level = ctx["sOBHigh"][i] + atr * 0.3
            elif (fvgS or bosS) and ctx["sActive"][i]:
                entry = ctx["sCE"][i]; sl_level = ctx["sTop"][i] + atr * 0.3
            elif ctx["sActive"][i]:
                entry = ctx["sCE"][i]; sl_level = ctx["bear_sweep_high"][i] + atr * 0.3
            else:
                entry = close; sl_level = ctx["bear_sweep_high"][i] + atr * 0.3
            direction = "SELL"

        if np.isnan(entry) or np.isnan(sl_level):
            return None
        sl_pts = abs(entry - sl_level) * MULT
        if sl_pts < MIN_SL_PTS or sl_pts > MAX_SL_PTS:
            return None
        sl_dist = abs(entry - sl_level)
        if direction == "BUY":
            tp1, tp2, tp3 = entry + sl_dist*TP1_R, entry + sl_dist*TP2_R, entry + sl_dist*TP3_R
        else:
            tp1, tp2, tp3 = entry - sl_dist*TP1_R, entry - sl_dist*TP2_R, entry - sl_dist*TP3_R
        print(f"   [check_signal] ✅ {direction} {model} @ {entry}")
        return {"model": model, "direction": direction, "entry": round(entry, 2), "sl": round(sl_level, 2),
                "tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2)}
    except Exception as e:
        print(f"   [check_signal] ❌ Exception: {e}")
        traceback.print_exc()
        return None


# ==================== الدالة الرئيسية ====================
def main():
    print("🤖 MSNR Bot Scan — بدء...")
    try:
        state = load_state()
        now_utc = datetime.now(ZoneInfo("UTC"))
        now_ny = now_utc.astimezone(NY_TZ)

        if now_ny.weekday() >= 5:
            day_name = "السبت" if now_ny.weekday() == 5 else "الأحد"
            print(f"📅 الويكند ({day_name}) — البوت معطّل")
            save_state(state)
            return

        today = now_ny.date().isoformat()
        if state["last_day"] != today:
            state["trade_count_today"] = 0
            state["models_done_today"] = []
            state["last_day"] = today

        print("📌 جلب M5...")
        df5 = fetch_candles(SYMBOL, "5min", 500)
        print("📌 جلب H1...")
        df1h = fetch_candles(SYMBOL, "1h", 500)

        if df5 is None or df1h is None:
            send_telegram("❌ فشل جلب البيانات من Twelve Data")
            save_state(state); return

        last_candle_time = df5.iloc[-1]["datetime"]
        minutes_old = (now_utc - last_candle_time).total_seconds() / 60
        print(f"⏱️ عمر آخر شمعة: {minutes_old:.1f} دقيقة")
        if minutes_old > STALE_DATA_MINUTES:
            print(f"⚠️ البيانات قديمة ({minutes_old:.0f} دقيقة) — تخطي")
            save_state(state)
            return

        price = float(df5.iloc[-1]["close"])
        print(f"💰 السعر: {price} | Mosul: {fmt_mosul(now_utc)} | NY: {now_ny.strftime('%H:%M')}")

        if state["active_trade"] is not None:
            print("📌 إدارة صفقة نشطة...")
            manage_trade(state, price, now_utc)
            save_state(state)
            print("✅ انتهى — تم إدارة الصفقة")
            return

        print("📌 فحص إشارة...")
        sig = check_signal(df5, df1h, state, now_utc, now_ny)
        if sig is None:
            print("✅ انتهى — لا إشارة")
            save_state(state); return

        model = sig["model"]
        if model in state["models_done_today"]:
            print(f"   [main] ⏭️ {model} مضروب اليوم — تخطي")
            save_state(state); return

        print("📌 إشارة موجودة، إرسال...")
        state["active_trade"] = {**sig, "tp1_hit": False, "tp2_hit": False, "tp3_hit": False, "entry_time": now_utc.isoformat()}
        state["trade_count_today"] += 1
        state["last_entry_time"] = now_utc.isoformat()
        state["models_done_today"].append(model)
        send_telegram(
            f"🚀 صفقة جديدة!\n\n"
            f"النموذج: {sig['model']}\n"
            f"الاتجاه: {sig['direction']}\n"
            f"الدخول: {sig['entry']}\n"
            f"الستوب: {sig['sl']}\n"
            f"TP1: {sig['tp1']}\n"
            f"TP2: {sig['tp2']}\n"
            f"TP3: {sig['tp3']}\n\n"
            f"الوقت: {fmt_mosul(now_utc)} (الموصل)"
        )
        save_state(state)
        print("✅ إشارة مرسلة")
    except Exception as e:
        print(f"❌❌❌ خطأ في main: {e}")
        traceback.print_exc()
        try:
            send_telegram(f"❌ MSNR Bot Scan خطأ: {e}")
        except:
            pass


if __name__ == "__main__":
    main()