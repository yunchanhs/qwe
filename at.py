import time
import pyupbit
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# API 키 설정
ACCESS_KEY = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
SECRET_KEY = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"

# 설정값
STOP_LOSS_THRESHOLD = -0.03  # -3% 손절
TAKE_PROFIT_THRESHOLD = 0.05  # +5% 익절
COOLDOWN_TIME = timedelta(minutes=5)
ML_THRESHOLD = 0.02  # 머신러닝 모델의 매수 신호 임계값

recent_trades = {}
entry_prices = {}

# 머신러닝 모델 생성
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 데이터 수집 및 지표 계산 함수
def get_features(ticker):
    """코인의 과거 데이터와 지표를 가져와 머신러닝에 적합한 피처 생성"""
    df = pyupbit.get_ohlcv(ticker, interval="minute5", count=200)
    df['macd'], df['signal'] = get_macd(ticker)
    df['rsi'] = get_rsi(ticker)
    df['adx'] = get_adx(ticker)
    df['atr'] = get_atr(ticker)
    df['return'] = df['close'].pct_change()
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1  # 다음 5분봉 수익률

    # NaN 값 제거
    df.dropna(inplace=True)
    return df

def train_model():
    """모든 코인의 데이터를 사용해 모델 학습"""
    tickers = pyupbit.get_tickers(fiat="KRW")
    data = []

    for ticker in tickers:
        try:
            features = get_features(ticker)
            data.append(features)
        except Exception as e:
            print(f"[{ticker}] 데이터 준비 중 에러 발생: {e}")

    # 데이터 통합
    full_data = pd.concat(data)
    X = full_data[['macd', 'signal', 'rsi', 'adx', 'atr', 'return']]
    y = full_data['future_return']

    # 학습-테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model.fit(X_train, y_train)

    # 모델 성능 평가
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"모델 학습 완료. 테스트 MSE: {mse:.6f}")

def get_ml_signal(ticker):
    """머신러닝 모델을 사용해 매수 신호 예측"""
    try:
        features = get_features(ticker)
        X_latest = features[['macd', 'signal', 'rsi', 'adx', 'atr', 'return']].iloc[-1:]
        prediction = model.predict(X_latest)[0]
        return prediction
    except Exception as e:
        print(f"[{ticker}] 머신러닝 신호 계산 중 에러 발생: {e}")
        return 0

# 매매 함수들 (기존 로직 유지)

# 메인 로직
if __name__ == "__main__":
    upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
    print("자동매매 시작!")

    # 머신러닝 모델 초기 학습
    train_model()

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

                    # 머신러닝 신호 계산
                    ml_signal = get_ml_signal(ticker)

                    # 기존 지표 계산
                    macd, signal = get_macd(ticker)
                    rsi = get_rsi(ticker)
                    adx = get_adx(ticker)
                    current_price = pyupbit.get_current_price(ticker)

                    # 매수 조건
                    if ml_signal > ML_THRESHOLD and macd > signal and rsi < 30 and adx > 25 and krw_balance > 5000:
                        buy_amount = krw_balance * 0.1
                        buy_result = buy_crypto_currency(ticker, buy_amount)
                        if buy_result:
                            entry_prices[ticker] = current_price
                            recent_trades[ticker] = now
                            print(f"[{ticker}] 매수 완료. 금액: {buy_amount:.2f}, 가격: {current_price:.2f}")

                    # 매도 조건 (손절/익절)
                    elif ticker in entry_prices:
                        entry_price = entry_prices[ticker]
                        change_ratio = (current_price - entry_price) / entry_price

                        if change_ratio <= STOP_LOSS_THRESHOLD or change_ratio >= TAKE_PROFIT_THRESHOLD:
                            coin_balance = get_balance(ticker.split('-')[1])
                            sell_result = sell_crypto_currency(ticker, coin_balance)
                            if sell_result:
                                recent_trades[ticker] = now
                                print(f"[{ticker}] 매도 완료. 잔고: {coin_balance:.4f}, 가격: {current_price:.2f}")

                except Exception as e:
                    print(f"[{ticker}] 처리 중 에러 발생: {e}")

            # 1분 대기
            time.sleep(60)

    except Exception as e:
        print(f"시스템 에러: {e}")
