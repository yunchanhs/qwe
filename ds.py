import time
import pyupbit
import datetime
import schedule
from requests.auth import AuthBase
import requests
from Prophet import Prophet

access = "J8iGqPwfjkX7Yg9bdzwFGkAZcTPU7rElXRozK7O4"
secret = "6MGxH2WjIftgQ85SLK1bcLxV4emYvrpbk6nYuqRN"
myToken = "xoxp-3889049319573-3885392999734-3895691939894-0c79ab2fc04c5aa3022b15520e075b70"



def post_message(token, channel, text):
    """슬랙 메시지 전송"""
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+token},
        data={"channel": channel,"text": text}
    )
    print(response)

def dbgout(message):
    """인자로 받은 문자열을 파이썬 셸과 슬랙으로 동시에 출력한다."""
    print(datetime.now().strftime('[%m/%d %H:%M:%S]'), message)
    strbuf = datetime.now().strftime('[%m/%d %H:%M:%S] ') + message
    post_message("xoxp-3889049319573-3885392999734-3895691939894-0c79ab2fc04c5aa3022b15520e075b70","#bit", strbuf)

def get_target_price(ticker, k):
    """변동성 돌파 전략으로 매수 목표가 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=2)
    target_price = df.iloc[0]['close'] + (df.iloc[0]['high'] - df.iloc[0]['low']) * k
    return target_price

def get_start_time(ticker):
    """시작 시간 조회"""
    df = pyupbit.get_ohlcv(ticker, interval="day", count=1)
    start_time = df.index[0]
    return start_time

def get_balance(ticker):
    """잔고 조회"""
    balances = upbit.get_balances()
    for b in balances:
        if b['currency'] == ticker:
            if b['balance'] is not None:
                return float(b['balance'])
            else:
                return 0
    return 0

def get_current_price(ticker):
    """현재가 조회"""
    return pyupbit.get_orderbook(ticker=ticker)["orderbook_units"][0]["ask_price"]

predicted_close_price = 0
def predict_price(ticker):
    """Prophet으로 당일 종가 가격 예측"""
    global predicted_close_price
    df = pyupbit.get_ohlcv(ticker, interval="minute60")
    df = df.reset_index()
    df['ds'] = df['index']
    df['y'] = df['close']
    data = df[['ds','y']]
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=24, freq='H')
    forecast = model.predict(future)
    closeDf = forecast[forecast['ds'] == forecast.iloc[-1]['ds'].replace(hour=9)]
    if len(closeDf) == 0:
        closeDf = forecast[forecast['ds'] == data.iloc[-1]['ds'].replace(hour=9)]
    closeValue = closeDf['yhat'].values[0]
    predicted_close_price = closeValue
predict_price("KRW-XRP")
schedule.every().hour.do(lambda: predict_price("KRW-XRP"))   

# 로그인
upbit = pyupbit.Upbit(access, secret)
print("autotrade start")
# 시작 메세지 슬랙 전송
post_message("xoxp-3889049319573-3885392999734-3895691939894-0c79ab2fc04c5aa3022b15520e075b70" ,"#bit", "autotrade start")

# 자동매매 시작
while True:
    try:
        now = datetime.datetime.now()
        start_time = get_start_time("KRW-XRP")
        end_time = start_time + datetime.timedelta(days=1)
        schedule.run_pending()

        if start_time < now < end_time - datetime.timedelta(seconds=10):
            target_price = get_target_price("KRW-XRP", 0.5)
            current_price = get_current_price("KRW-XRP")
            if target_price < current_price and current_price < predicted_close_price:
                krw = get_balance("KRW")
                if krw > 5000:
                    buy_result = upbit.buy_market_order("KRW-XRP", krw*0.9995)
                    post_message("xoxp-3889049319573-3885392999734-3895691939894-0c79ab2fc04c5aa3022b15520e075b70", "#bit", "XRP buy : " +str(buy_result))

        else:
            btc = get_balance("XRP")
            if btc > 0.00008:
                sell_result = upbit.sell_market_order("KRW-XRP", btc*0.9995)
                post_message("xoxp-3889049319573-3885392999734-3895691939894-0c79ab2fc04c5aa3022b15520e075b70", "#bit", "XRP buy : " +str(sell_result)) 
        time.sleep(1)
   
    except Exception as e:
        print(e)
        post_message("xoxp-3889049319573-3885392999734-3895691939894-0c79ab2fc04c5aa3022b15520e075b70", "#bit", e)
        time.sleep(1)   
