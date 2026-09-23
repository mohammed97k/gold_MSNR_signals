import json
import math
import os
import time
from datetime import datetime, timezone
import zoneinfo
import pandas as pd
import requests

# =====================================================
# الإعدادات وقنوات التليجرام
# =====================================================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
TWELVE_DATA_API_KEY = os.environ.get("TWELVE_DATA_API_KEY", "f66d01dd648c41898a1d908f17fff5a0")
STATE_FILE = "state.json"

# التوقيت المحلي للموصل / العراق
MOSUL_TZ = zoneinfo.ZoneInfo("Asia/Baghdad")

# ملف تعريف الأصول والحدود للعملات الأربعة
PROFILES = {
    "GBPUSD": {
        "symbol": "GBP/USD",
        "profile": "FOREX",
        "digits": 5,
        "point_mult": 10000.0,
        "min_sl": 0.0008,   # 8 نقاط
        "max_sl": 0.0025,   # 25 نقطة كحد أقصى
        "min_tp": 0.0020,   # 20 نقطة
        "max_tp": 0.0060,   # 60 نقطة
        "max_drift": 0.0012 # 12 نقطة أقصى انزلاق
    },
    "USDJPY": {
        "symbol": "USD/JPY",
        "profile": "FOREX_JPY",
        "digits": 3,
        "point_mult": 100.0,
        "min_sl": 0.08,     # 8 نقاط
        "max_sl": 0.25,     # 25 نقطة كحد أقصى
        "min_tp": 0.20,     # 20 نقطة
        "max_tp": 0.60,     # 60 نقطة
        "max_drift": 0.12   # 12 نقطة أقصى انزلاق
    },
    "AUDUSD": {
        "symbol": "AUD/USD",
        "profile": "FOREX",
        "digits": 5,
        "point_mult": 10000.0,
        "min_sl": 0.0008,   # 8 نقاط
        "max_sl": 0.0025,   # 25 نقطة كحد أقصى
        "min_tp": 0.0018,   # 18 نقطة
        "max_tp": 0.0050,   # 50 نقطة
        "max_drift": 0.0010 # 10 نقاط أقصى انزلاق
    },
    "USDCAD": {
        "symbol": "USD/CAD",
        "profile": "FOREX",
        "digits": 5,
        "point_mult": 10000.0,
        "min_sl": 0.0008,   # 8 نقاط
        "max_sl": 0.0025,   # 25 نقطة كحد أقصى
        "min_tp": 0.0018,   # 18 نقطة
        "max_tp": 0.0050,   # 50 نقطة
        "max_drift": 0.0010 # 10 نقاط أقصى انزلاق
    }
}

MAX_TRADES_PER_DAY = 3
HTF_EMA_LEN = 50
SWING_LEN = 5
FVG_MIN_ATR_PCT = 0.15
DISP_MULT = 1.5
ZONE_MAX_AGE = 30


# =====================================================
# دالة إرسال التيليجرام
# =====================================================
def send_telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram tokens are missing!")
        print(msg)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, json=payload, timeout=15)
        if not r.ok:
            print(f"Telegram error: {r.text}")
    except Exception as e:
        print(f"Failed to send telegram message: {e}")


# =====================================================
# إدارة الحالات (State Management)
# =====================================================
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


# =====================================================
# جلب البيانات من Twelve Data
# =====================================================
def fetch_twelve_data(symbol: str, interval: str, outputsize: int = 100):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
    }
    try:
        res = requests.get(url, params=params, timeout=20)
        data = res.json()
        if "values" not in data:
            print(f"خطأ جلب بيانات {symbol} ({interval}): {data.get('message', data)}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        
        df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close"
        }, inplace=True)
        return df
    except Exception as e:
        print(f"استثناء أثناء طلب Twelve Data لـ {symbol}: {e}")
        return pd.DataFrame()


# =====================================================
# الحسابات الفنية
# =====================================================
def calc_atr(df: pd.DataFrame, period: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"].shift(1)
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# =====================================================
# محرك الاستراتيجية
# =====================================================
def process_symbol(symbol_name: str, cfg: dict, state: dict, now_local: datetime):
    today_str = now_local.strftime("%Y-%m-%d")
    sym_state = state.get(
        symbol_name, {"active_trade": None, "trades_today": 0, "last_date": today_str}
    )

    if sym_state.get("last_date") != today_str:
        sym_state["trades_today"] = 0
        sym_state["last_date"] = today_str

    sym = cfg["symbol"]
    digits = cfg["digits"]
    mult = cfg["point_mult"]
    max_drift = cfg.get("max_drift", 0.0010)

    # جلب بيانات فريم الساعة وفريم اليومي
    df_h1 = fetch_twelve_data(sym, "1h", 120)
    time.sleep(8)

    df_d = fetch_twelve_data(sym, "1day", 80)
    time.sleep(8)

    if df_h1.empty or df_d.empty or len(df_h1) < 40 or len(df_d) < HTF_EMA_LEN:
        print(f"[{symbol_name}] البيانات غير كافية أو تعذر الوصول للمزود.")
        return

    # التحيز اليومي HTF Bias
    df_d["EMA50"] = df_d["Close"].ewm(span=HTF_EMA_LEN, adjust=False).mean()
    last_d_close = df_d["Close"].iloc[-1]
    last_d_ema = df_d["EMA50"].iloc[-1]
    pd_high = df_d["High"].iloc[-2]
    pd_low = df_d["Low"].iloc[-2]

    htf_bias_up = last_d_close > last_d_ema
    htf_bias_down = last_d_close < last_d_ema

    df_h1["ATR14"] = calc_atr(df_h1, 14)

    # 1. متابعة الصفقات المفتوحة لحظياً (ضرب الستوب أو الهدف)
    active_trade = sym_state.get("active_trade")
    if active_trade is not None:
        live_candle = df_h1.iloc[-1]
        c_high = float(live_candle["High"])
        c_low = float(live_candle["Low"])

        side = active_trade["side"]
        entry = active_trade["entry"]
        sl = active_trade["sl"]
        tp = active_trade["tp"]
        rr = active_trade["rr"]
        entry_time = active_trade["entry_time"]

        hit_tp = (c_high >= tp) if side == "buy" else (c_low <= tp)
        hit_sl = (c_low <= sl) if side == "buy" else (c_high >= sl)

        if hit_tp or hit_sl:
            outcome = "🎯 ضرب الهدف (TP)" if hit_tp else "🛑 ضرب الستوب (SL)"
            exit_price = tp if hit_tp else sl
            move_pts = abs(exit_price - entry) * mult
            exit_time_str = now_local.strftime("%I:%M %p")

            exit_msg = (
                f"<b>🚨 إغلاق صفقة — {symbol_name}</b>\n"
                f"<b>النتيجة:</b> {outcome}\n"
                f"<b>النوع:</b> {side.upper()}\n"
                f"<b>سعر الدخول:</b> {entry:.{digits}f}\n"
                f"<b>سعر الخروج:</b> {exit_price:.{digits}f}\n"
                f"<b>النقاط:</b> {move_pts:.1f} نقطة\n"
                f"<b>R:R:</b> 1:{rr:.2f}\n"
                f"<b>وقت الدخول:</b> {entry_time}\n"
                f"<b>وقت الخروج:</b> {exit_time_str} (توقيت الموصل)"
            )
            send_telegram(exit_msg)
            sym_state["active_trade"] = None
            state[symbol_name] = sym_state
            return

    # 2. فحص إمكانية فتح صفقة جديدة
    if (
        sym_state.get("active_trade") is not None
        or sym_state.get("trades_today", 0) >= MAX_TRADES_PER_DAY
    ):
        state[symbol_name] = sym_state
        return

    # حساب الهيكل والـ FVG والـ OB
    total_candles = len(df_h1)
    confirmed_candles_count = total_candles - 1

    last_swing_high = None
    prev_swing_high = None
    last_swing_low = None
    prev_swing_low = None
    structure_trend = 0

    bull_fvg = None
    bear_fvg = None
    bull_ob = None
    bear_ob = None
    bull_breaker = None
    bear_breaker = None

    for i in range(SWING_LEN * 2, confirmed_candles_count):
        window_high = df_h1["High"].iloc[i - SWING_LEN * 2 : i + 1]
        mid_idx = i - SWING_LEN
        if df_h1["High"].iloc[mid_idx] == window_high.max():
            prev_swing_high = last_swing_high
            last_swing_high = float(df_h1["High"].iloc[mid_idx])

        window_low = df_h1["Low"].iloc[i - SWING_LEN * 2 : i + 1]
        if df_h1["Low"].iloc[mid_idx] == window_low.min():
            prev_swing_low = last_swing_low
            last_swing_low = float(df_h1["Low"].iloc[mid_idx])

        c_close = df_h1["Close"].iloc[i]
        c_open = df_h1["Open"].iloc[i]
        c_high = df_h1["High"].iloc[i]
        c_low = df_h1["Low"].iloc[i]
        c_atr = df_h1["ATR14"].iloc[i]

        p_close = df_h1["Close"].iloc[i - 1]
        p_open = df_h1["Open"].iloc[i - 1]
        p_high = df_h1["High"].iloc[i - 1]
        p_low = df_h1["Low"].iloc[i - 1]

        if last_swing_high and c_close > last_swing_high and structure_trend <= 0:
            structure_trend = 1
        elif last_swing_low and c_close < last_swing_low and structure_trend >= 0:
            structure_trend = -1

        if i >= 2 and not math.isnan(c_atr):
            if c_low > df_h1["High"].iloc[i - 2] and (c_low - df_h1["High"].iloc[i - 2]) > (c_atr * FVG_MIN_ATR_PCT):
                bull_fvg = {"top": c_low, "bot": df_h1["High"].iloc[i - 2], "bar": i}
            if c_high < df_h1["Low"].iloc[i - 2] and (df_h1["Low"].iloc[i - 2] - c_high) > (c_atr * FVG_MIN_ATR_PCT):
                bear_fvg = {"top": df_h1["Low"].iloc[i - 2], "bot": c_high, "bar": i}

        candle_range = c_high - c_low
        is_disp = not math.isnan(c_atr) and candle_range > (c_atr * DISP_MULT)

        if is_disp and c_close > c_open and p_close < p_open:
            bull_ob = {"top": p_high, "bot": p_low, "bar": i}
        if is_disp and c_close < c_open and p_close > p_open:
            bear_ob = {"top": p_high, "bot": p_low, "bar": i}

        if bull_ob and c_close < bull_ob["bot"]:
            bear_breaker = {"top": bull_ob["top"], "bot": bull_ob["bot"], "bar": i}
            bull_ob = None

        if bear_ob and c_close > bear_ob["top"]:
            bull_breaker = {"top": bear_ob["top"], "bot": bear_ob["bot"], "bar": i}
            bear_ob = None

        if bull_fvg and (c_close < bull_fvg["bot"] or (confirmed_candles_count - bull_fvg["bar"]) > ZONE_MAX_AGE):
            bull_fvg = None
        if bear_fvg and (c_close > bear_fvg["top"] or (confirmed_candles_count - bear_fvg["bar"]) > ZONE_MAX_AGE):
            bear_fvg = None
        if bull_ob and (confirmed_candles_count - bull_ob["bar"]) > ZONE_MAX_AGE:
            bull_ob = None
        if bear_ob and (confirmed_candles_count - bear_ob["bar"]) > ZONE_MAX_AGE:
            bear_ob = None
        if bull_breaker and (confirmed_candles_count - bull_breaker["bar"]) > ZONE_MAX_AGE:
            bull_breaker = None
        if bear_breaker and (confirmed_candles_count - bear_breaker["bar"]) > ZONE_MAX_AGE:
            bear_breaker = None

    live_candle = df_h1.iloc[-1]
    live_h = float(live_candle["High"])
    live_l = float(live_candle["Low"])
    current_price = float(live_candle["Close"])

    in_bull_zone = False
    bull_entry = 0.0
    if bull_ob and live_l <= bull_ob["top"] and live_h >= bull_ob["bot"]:
        in_bull_zone = True
        bull_entry = (bull_ob["top"] + bull_ob["bot"]) / 2.0
    elif bull_fvg and live_l <= bull_fvg["top"] and live_h >= bull_fvg["bot"]:
        in_bull_zone = True
        bull_entry = (bull_fvg["top"] + bull_fvg["bot"]) / 2.0
    elif bull_breaker and live_l <= bull_breaker["top"] and live_h >= bull_breaker["bot"]:
        in_bull_zone = True
        bull_entry = (bull_breaker["top"] + bull_breaker["bot"]) / 2.0

    in_bear_zone = False
    bear_entry = 0.0
    if bear_ob and live_l <= bear_ob["top"] and live_h >= bear_ob["bot"]:
        in_bear_zone = True
        bear_entry = (bear_ob["top"] + bear_ob["bot"]) / 2.0
    elif bear_fvg and live_l <= bear_fvg["top"] and live_h >= bear_fvg["bot"]:
        in_bear_zone = True
        bear_entry = (bear_fvg["top"] + bear_fvg["bot"]) / 2.0
    elif bear_breaker and live_l <= bear_breaker["top"] and live_h >= bear_breaker["bot"]:
        in_bear_zone = True
        bear_entry = (bear_breaker["top"] + bear_breaker["bot"]) / 2.0

    bull_signal = htf_bias_up and structure_trend != -1 and in_bull_zone
    bear_signal = htf_bias_down and structure_trend != 1 and in_bear_zone

    # تنفيذ شراء
    if bull_signal:
        drift = abs(current_price - bull_entry)
        if drift <= max_drift:
            nat_sl = (bull_entry - last_swing_low) if last_swing_low is not None else cfg["min_sl"]
            sl_dist = max(min(nat_sl, cfg["max_sl"]), cfg["min_sl"])
            sl_final = bull_entry - sl_dist

            if prev_swing_high and prev_swing_high > bull_entry:
                nat_tp = prev_swing_high - bull_entry
            elif pd_high and pd_high > bull_entry:
                nat_tp = pd_high - bull_entry
            else:
                nat_tp = cfg["max_tp"]

            tp_dist = max(min(nat_tp, cfg["max_tp"]), cfg["min_tp"])
            tp_final = bull_entry + tp_dist
            rr = tp_dist / sl_dist

            entry_time_str = now_local.strftime("%I:%M %p")
            msg = (
                f"<b>🟢 إشارة دخول فورية — {symbol_name}</b>\n"
                f"<b>الاتجاه:</b> شراء (BUY)\n"
                f"<b>نقطة الدخول (المنطقة):</b> {bull_entry:.{digits}f}\n"
                f"<b>السعر الحالي:</b> {current_price:.{digits}f}\n"
                f"<b>الستوب (SL):</b> {sl_final:.{digits}f}\n"
                f"<b>الهدف (TP):</b> {tp_final:.{digits}f}\n"
                f"<b>العائد للمخاطرة R:R:</b> 1:{rr:.2f}\n"
                f"<b>وقت الدخول:</b> {entry_time_str} (توقيت الموصل)\n"
                f"⚠️ إدارة رأس المال أولاً!"
            )
            send_telegram(msg)

            sym_state["active_trade"] = {
                "side": "buy",
                "entry": bull_entry,
                "sl": sl_final,
                "tp": tp_final,
                "rr": rr,
                "entry_time": entry_time_str,
            }
            sym_state["trades_today"] = sym_state.get("trades_today", 0) + 1

    # تنفيذ بيع
    elif bear_signal:
        drift = abs(current_price - bear_entry)
        if drift <= max_drift:
            nat_sl = (last_swing_high - bear_entry) if last_swing_high is not None else cfg["min_sl"]
            sl_dist = max(min(nat_sl, cfg["max_sl"]), cfg["min_sl"])
            sl_final = bear_entry + sl_dist

            if prev_swing_low and prev_swing_low < bear_entry:
                nat_tp = bear_entry - prev_swing_low
            elif pd_low and pd_low < bear_entry:
                nat_tp = bear_entry - pd_low
            else:
                nat_tp = cfg["max_tp"]

            tp_dist = max(min(nat_tp, cfg["max_tp"]), cfg["min_tp"])
            tp_final = bear_entry - tp_dist
            rr = tp_dist / sl_dist

            entry_time_str = now_local.strftime("%I:%M %p")
            msg = (
                f"<b>🔴 إشارة دخول فورية — {symbol_name}</b>\n"
                f"<b>الاتجاه:</b> بيع (SELL)\n"
                f"<b>نقطة الدخول (المنطقة):</b> {bear_entry:.{digits}f}\n"
                f"<b>السعر الحالي:</b> {current_price:.{digits}f}\n"
                f"<b>الستوب (SL):</b> {sl_final:.{digits}f}\n"
                f"<b>الهدف (TP):</b> {tp_final:.{digits}f}\n"
                f"<b>العائد للمخاطرة R:R:</b> 1:{rr:.2f}\n"
                f"<b>التوقيت:</b> {entry_time_str} (توقيت الموصل)\n"
                f"⚠️ إدارة رأس المال أولاً!"
            )
            send_telegram(msg)

            sym_state["active_trade"] = {
                "side": "sell",
                "entry": bear_entry,
                "sl": sl_final,
                "tp": tp_final,
                "rr": rr,
                "entry_time": entry_time_str,
            }
            sym_state["trades_today"] = sym_state.get("trades_today", 0) + 1

    state[symbol_name] = sym_state


# =====================================================
# نقطة التشغيل الرئيسية
# =====================================================
def main():
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(MOSUL_TZ)

    state = load_state()

    for name, cfg in PROFILES.items():
        try:
            print(f"جاري فحص {name}...")
            process_symbol(name, cfg, state, now_local)
        except Exception as e:
            print(f"[{name}] خطأ أثناء المعالجة: {e}")

    save_state(state)


if __name__ == "__main__":
    main()
