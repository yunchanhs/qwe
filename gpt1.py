import time
import pyupbit
import pandas as pd
from datetime import datetime, timedelta

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 손절 및 익절 비율 설정
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절

# 쿨다운 타임 설정
COOLDOWN_TIME = timedelta(minutes=5)

# 최근 매매 기록 저장 (쿨다운 타임 관리)
recent_trades = {}

# 진입가 저장
entry_prices = {}

# MACD 계산 함수
def get_macd(ticker, short_window=12, long_window=26, signal_window=9):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    short_ema = df['close'].ewm(span=short_window).mean()
    long_ema = df['close'].ewm(span=long_window).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window).mean()
    return macd.iloc[-1], signal.iloc[-1]

# RSI 계산 함수
def get_rsi(ticker, period=14):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# ADX 계산 함수
def get_adx(ticker, window=14):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff().clip(lower=0)
    minus_dm = -low.diff().clip(upper=0)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)

    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx.iloc[-1]

# EMA 계산 함수
def get_ema(ticker, window=20):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    ema = df['close'].ewm(span=window).mean()
    return ema.iloc[-1]

# ATR 계산 함수
def get_atr(ticker, window=14):
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    high = df['high']
    low = df['low']
    close = df['close']

    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr.iloc[-1]

# 매수 함수
def buy_crypto_currency(ticker, amount):
    try:
        result = upbit.buy_market_order(ticker, amount)
        return result
    except Exception as e:
        print(f"[{ticker}] 매수 에러: {e}")
        return None

# 매도 함수
def sell_crypto_currency(ticker, amount):
    try:
        result = upbit.sell_market_order(ticker, amount)
        return result
    except Exception as e:
        print(f"[{ticker}] 매도 에러: {e}")
        return None

# 잔고 확인 함수
def get_balance(ticker):
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            return float(b.get('balance', 0))
    return 0

# 로그 기록 함수
def log_trade(action, ticker, amount, price):
    with open("trade_log.txt", "a") as f:
        f.write(f"{datetime.now()}, {action}, {ticker}, {amount}, {price}\n")

def log_signal(ticker, macd, rsi, adx, ema, atr, action):
    with open("signals_log.txt", "a") as f:
        f.write(f"{datetime.now()}, {ticker}, MACD: {macd:.2f}, RSI: {rsi:.2f}, ADX: {adx:.2f}, EMA: {ema:.2f}, ATR: {atr:.2f}, Action: {action}\n")

# 메인 로직
if __name__ == "__main__":
    # 업비트 API 객체 생성
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

    print("자동매매 시작!")
    try:
        while True:
            tickers = pyupbit.get_tickers(fiat="KRW")
            krw_balance = get_balance("KRW")

            for ticker in tickers:
                try:
                    now = datetime.now()

                    # 쿨다운 타임 체크
                    if ticker in recent_trades and now - recent_trades[ticker] < COOLDOWN_TIME:
                        continue

                    # 각 지표 계산
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    adx = get_adx(ticker)
                    ema = get_ema(ticker)
                    atr = get_atr(ticker)

                    coin_balance = get_balance(ticker.split('-')[1])
                    current_price = pyupbit.get_current_price(ticker)

                    # 매수 조건 (MACD 크로스, RSI, ADX, EMA, ATR)
                    if macd > signal and rsi < 30 and adx > 25 and ema > current_price and atr > 0.5 and krw_balance > 5000:
                        buy_amount = krw_balance * 0.1  # 잔고의 10% 매수
                        buy_result = buy_crypto_currency(ticker, buy_amount)
                        if buy_result:
                            entry_prices[ticker] = current_price  # 진입가 저장
                            recent_trades[ticker] = now
                            log_trade("BUY", ticker, buy_amount, current_price)
                            log_signal(ticker, macd, rsi, adx, ema, atr, "BUY")
                            print(f"[{ticker}] 매수 완료. 금액: {buy_amount:.2f}, 가격: {current_price:.2f}")

                    # 매도 조건 (손절/익절)
                    elif ticker in entry_prices:
                        entry_price = entry_prices[ticker]
                        change_ratio = (current_price - entry_price) / entry_price

                        # 손절 또는 익절
                        if change_ratio <= STOP_LOSS_THRESHOLD or change_ratio >= TAKE_PROFIT_THRESHOLD:
                            sell_result = sell_crypto_currency(ticker, coin_balance)
                            if sell_result:
                                recent_trades[ticker] = now
                                log_trade("SELL", ticker, coin_balance, current_price)
                                log_signal(ticker, macd, rsi, adx, ema, atr, "SELL")
                                print(f"[{ticker}] 매도 완료. 잔고: {coin_balance:.4f}, 가격: {current_price:.2f}")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

            # 1분 대기
            time.sleep(60)

    except Exception as e:
        print(f"시스템 에러: {e}")
