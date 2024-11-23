import time
import pyupbit
import pandas as pd

# 업비트 API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"  # 발급받은 API access key 입력
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"  # 발급받은 API secret key 입력

# MACD 계산 함수
def get_macd(ticker, short_window=12, long_window=26, signal_window=9):
    """MACD와 Signal 계산"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    short_ema = df['close'].ewm(span=short_window).mean()
    long_ema = df['close'].ewm(span=long_window).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window).mean()
    return macd.iloc[-1], signal.iloc[-1]

# RSI 계산 함수
def get_rsi(ticker, period=14):
    """RSI 계산"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5")
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]

# 잔고 확인 함수
def get_balance(ticker):
    """잔고 조회"""
    try:
        balances = upbit.get_balances()
        if isinstance(balances, list):
            for b in balances:
                if b['currency'] == ticker:
                    return float(b.get('balance', 0))
        else:
            print("get_balances() 반환 값이 리스트가 아닙니다. 데이터:", balances)
    except Exception as e:
        print(f"get_balance() 처리 중 에러 발생: {e}")
    return 0

# 매수 함수
def buy_crypto_currency(ticker, amount):
    """시장가로 지정된 금액만큼 매수"""
    return upbit.buy_market_order(ticker, amount)

# 매도 함수
def sell_crypto_currency(ticker, amount):
    """시장가로 지정된 수량만큼 매도"""
    return upbit.sell_market_order(ticker, amount)

# 메인 로직
if __name__ == "__main__":
    # 업비트 API 객체 생성
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

    # RSI 및 MACD 기준
    rsi_threshold_buy = 30   # RSI 과매도 기준
    rsi_threshold_sell = 70  # RSI 과매수 기준

    print("자동매매 시작!")

    try:
        while True:
            # 모든 KRW 마켓 코인 가져오기
            tickers = pyupbit.get_tickers(fiat="KRW")
            
            for ticker in tickers:
                try:
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    krw_balance = get_balance("KRW")  # 원화 잔고 확인
                    coin_balance = get_balance(ticker.split('-')[1])  # 해당 코인의 잔고 확인

                    print(f"[{ticker}] MACD: {macd:.2f}, Signal: {signal:.2f}, RSI: {rsi:.2f}")

                    # 매수 조건: MACD > Signal and RSI < 30
                    if macd > signal and rsi < rsi_threshold_buy and krw_balance > 5000:
                        print(f"[{ticker}] 매수 조건 충족. 매수 진행...")
                        buy_crypto_currency(ticker, krw_balance * 0.1)  # 전체 잔고의 10% 매수

                    # 매도 조건: MACD < Signal and RSI > 70
                    elif macd < signal and rsi > rsi_threshold_sell and coin_balance > 0.0001:
                        print(f"[{ticker}] 매도 조건 충족. 매도 진행...")
                        sell_crypto_currency(ticker, coin_balance)

                except Exception as e:
                    print(f"[{ticker}] 데이터 처리 중 에러 발생: {e}")
            
            # 60초 대기 (API 호출 제한 고려)
            time.sleep(60)

    except Exception as e:
        print(f"전체 시스템 에러 발생: {e}")
