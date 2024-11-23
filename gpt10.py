import time
import pyupbit
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# API 키 설정
ACCESS_KEY = "your-access-key"
SECRET_KEY = "your-secret-key"

# 매수/매도 조건 설정
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절
COOLDOWN_TIME = timedelta(minutes=5)

# 상위 코인 캐싱 설정
cached_tickers = []
last_ticker_update = None
TICKER_UPDATE_INTERVAL = timedelta(hours=1)

# 매매 기록 및 진입가 관리
recent_trades = {}
entry_prices = {}

# 업비트 객체 생성
upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

# 주요 코인 필터링 함수
def fetch_volume(ticker):
    """거래량 가져오기"""
    try:
        df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
        return ticker, df['volume'].iloc[-1] if df is not None else 0
    except:
        return ticker, 0

def get_major_tickers():
    """거래량 상위 코인 조회 및 캐싱"""
    global cached_tickers, last_ticker_update
    now = datetime.now()

    if last_ticker_update is None or now - last_ticker_update > TICKER_UPDATE_INTERVAL:
        try:
            all_tickers = pyupbit.get_tickers(fiat="KRW")
            with ThreadPoolExecutor(max_workers=10) as executor:
                volumes = list(executor.map(fetch_volume, all_tickers))

            # 거래량 상위 10개 코인 선택
            sorted_tickers = sorted(volumes, key=lambda x: x[1], reverse=True)
            cached_tickers = [ticker for ticker, _ in sorted_tickers[:10]]
            last_ticker_update = now
            print(f"[{now}] 상위 코인 목록 업데이트 완료: {cached_tickers}")

        except Exception as e:
            print(f"코인 목록 조회 실패: {e}")

    return cached_tickers

# 지표 계산 함수
def calculate_indicators(ticker):
    """MACD, RSI, ADX, EMA, ATR 계산"""
    try:
        df = pyupbit.get_ohlcv(ticker, interval="minute5")
        if df is None:
            return None

        # MACD
        short_ema = df['close'].ewm(span=12).mean()
        long_ema = df['close'].ewm(span=26).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # ADX
        high = df['high']
        low = df['low']
        close = df['close']
        plus_dm = high.diff().clip(lower=0)
        minus_dm = -low.diff().clip(upper=0)
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=14).mean()

        # EMA
        ema = df['close'].ewm(span=20).mean()

        # ATR
        atr = tr.rolling(window=14).mean()

        return {
            "macd": macd.iloc[-1],
            "signal": signal.iloc[-1],
            "rsi": rsi.iloc[-1],
            "adx": adx.iloc[-1],
            "ema": ema.iloc[-1],
            "atr": atr.iloc[-1],
        }
    except Exception as e:
        print(f"[{ticker}] 지표 계산 오류: {e}")
        return None

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

# 매수 조건 확인
def should_buy(indicators, current_price):
    return (
        indicators["macd"] > indicators["signal"]
        and indicators["rsi"] < 30
        and indicators["adx"] > 25
        and indicators["ema"] > current_price
        and indicators["atr"] > 0.5
    )

# 매도 조건 확인
def should_sell(ticker, current_price):
    entry_price = entry_prices.get(ticker, None)
    if entry_price:
        change_ratio = (current_price - entry_price) / entry_price
        return change_ratio <= STOP_LOSS_THRESHOLD or change_ratio >= TAKE_PROFIT_THRESHOLD
    return False

# 메인 실행 함수
if __name__ == "__main__":
    print("자동매매 시작!")

    try:
        while True:
            tickers = get_major_tickers()
            krw_balance = get_balance("KRW")

            # 병렬로 지표 계산
            with ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(calculate_indicators, tickers))

            for ticker, indicators in zip(tickers, results):
                if indicators is None:
                    continue

                now = datetime.now()
                if ticker in recent_trades and now - recent_trades[ticker] < COOLDOWN_TIME:
                    continue

                coin_balance = get_balance(ticker.split('-')[1])
                current_price = pyupbit.get_current_price(ticker)

                # 매수 조건
                if should_buy(indicators, current_price) and krw_balance > 5000:
                    buy_amount = krw_balance * 0.05
                    buy_result = buy_crypto_currency(ticker, buy_amount)
                    if buy_result:
                        entry_prices[ticker] = current_price
                        recent_trades[ticker] = now
                        print(f"[{ticker}] 매수 완료: {buy_amount:.2f}원")

                # 매도 조건
                elif should_sell(ticker, current_price):
                    sell_result = sell_crypto_currency(ticker, coin_balance)
                    if sell_result:
                        recent_trades[ticker] = now
                        print(f"[{ticker}] 매도 완료: {coin_balance:.4f}개")

            # 1분 대기
            time.sleep(60)

    except Exception as e:
        print(f"시스템 에러: {e}")
