import pyupbit
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import matplotlib.pyplot as plt

# 업비트 API 연결
access_key = 'J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4'
secret_key = '6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN'
upbit = pyupbit.Upbit(access_key, secret_key)

# 모든 마켓 조회 (코인 목록)
def get_all_markets():
    markets = pyupbit.get_market_all()
    return [market['market'] for market in markets if market['market'].startswith('KRW-')]  # KRW 마켓만

# 각 코인에 대한 데이터를 수집
def get_coin_data(coin, interval="day", count=1000):
    df = pyupbit.get_ohlcv(coin, interval=interval, count=count)
    return df['close'].values  # 종가 데이터만 사용

# LSTM 모델 훈련을 위한 데이터 전처리
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])  # 60일 동안의 데이터
        y.append(dataset[i + time_step, 0])  # 그 다음 날 가격
    return np.array(X), np.array(y)

# LSTM 모델 구성
def build_lstm_model(X):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        tf.keras.layers.LSTM(units=50, return_sequences=False),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 실시간 거래를 위한 환경 정의
class TradingEnv(gym.Env):
    def __init__(self, data, model, window_size=60):
        super(TradingEnv, self).__init__()
        self.data = data
        self.model = model
        self.window_size = window_size
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 행동: 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.window_size, 1), dtype=np.float32)
        self.balance = 100000  # 시작 잔고 (원화)
        self.btc_balance = 0  # 비트코인 보유량

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        return self.data[self.current_step - self.window_size: self.current_step].reshape(-1, 1)

    def step(self, action):
        current_price = self.data[self.current_step]
        self.current_step += 1
        next_price = self.data[self.current_step]

        reward = 0
        if action == 1:  # 매수
            if self.balance >= current_price:
                self.btc_balance += self.balance / current_price
                self.balance = 0  # 매수 후 현금 잔고 0으로 설정
                reward = 1  # 매수 후 가격 상승 시 보상
        elif action == 2:  # 매도
            if self.btc_balance > 0:
                self.balance += self.btc_balance * current_price
                self.btc_balance = 0  # 매도 후 비트코인 잔고 0으로 설정
                reward = 1  # 매도 후 가격 하락 시 보상

        done = self.current_step >= len(self.data) - 1
        next_state = self.data[self.current_step - self.window_size: self.current_step].reshape(-1, 1)
        return next_state, reward, done, {}

    def render(self):
        print(f"Balance: {self.balance}, BTC Balance: {self.btc_balance}")

# 리스크 관리 (Stop Loss, Take Profit)
STOP_LOSS = 0.02  # 2% 손실 발생 시 자동 매도
TAKE_PROFIT = 0.05  # 5% 이익 발생 시 자동 매도

def risk_management(current_price, action):
    if action == 1 and current_price < (current_price * (1 - STOP_LOSS)):  # Stop Loss
        return 2  # 매도
    elif action == 2 and current_price > (current_price * (1 + TAKE_PROFIT)):  # Take Profit
        return 2  # 매도
    return action

# 학습을 위한 LSTM 모델 훈련
def train_lstm_model(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    lstm_model = build_lstm_model(X)
    lstm_model.fit(X, y, epochs=10, batch_size=32)
    return lstm_model, scaler

# 강화학습 환경 설정
def setup_ppo_model(data, lstm_model):
    env = TradingEnv(data=data, model=lstm_model)
    env = DummyVecEnv([lambda: env])

    ppo_model = PPO("MlpPolicy", env, verbose=1)
    ppo_model.learn(total_timesteps=500000)
    return ppo_model, env

# 모든 코인에 대해 매매 수행
def trade_all_coins():
    markets = get_all_markets()  # 거래 가능한 모든 코인 목록 조회
    for market in markets:
        print(f"Trading on {market}")
        data = get_coin_data(market)
        lstm_model, scaler = train_lstm_model(data)
        
        # 강화학습 모델 학습
        ppo_model, env = setup_ppo_model(data, lstm_model)
        
        # 모델 테스트
        obs = env.reset()
        for _ in range(1000):
            action, _states = ppo_model.predict(obs)
            action = risk_management(obs[0][0][0], action)  # 리스크 관리 적용
            obs, rewards, dones, info = env.step(action)
            if dones:
                break
            env.render()

        # 실시간 거래 예시
        execute_trade(market, lstm_model, scaler)

# 실시간 거래 예시
def execute_trade(market, lstm_model, scaler):
    current_price = pyupbit.get_current_price(market)
    balance = upbit.get_balance("KRW")
    
    # 예측된 가격과 실시간 가격 비교하여 매매 결정
    predicted_price = lstm_model.predict(scaler.transform(current_price[-60:].reshape(1, -1)))[0][0]
    
    if predicted_price > current_price:  # 매수 신호
        if balance > 1000:  # 최소 거래 가능 금액
            upbit.buy_market_order(market, balance * 0.1)  # 10% 비율로 매수
    elif predicted_price < current_price:  # 매도 신호
        if upbit.get_balance(market) > 0.001:  # 최소 코인 매도 금액
            upbit.sell_market_order(market, upbit.get_balance(market))

# 실행 예시
trade_all_coins()
