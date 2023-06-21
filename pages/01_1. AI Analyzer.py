import streamlit as st
from datetime import datetime
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import pandas_datareader as data
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import optuna
import pandas_ta as ta
import ta as tas
from ta.volatility import BollingerBands
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from backtesting import Backtest, Strategy

import plotly.graph_objects as go

st.set_page_config(page_title = "STOCK ANALYZER", layout="wide")
hide_st_style = """
            <style>
            footer {visibility: visible;}
            footer:after {content:'Copyright @ AI Trader X'; display:block; position: relative;}
            </style>
            """
st.subheader("XGBoost & Backtesting & Optimiation") 
st.markdown(hide_st_style, unsafe_allow_html=True)
st.sidebar.subheader("STOCK ANALYZER TOOL V0.0.1")


All_Create = st.sidebar.checkbox("All")
AIOpt = st.sidebar.checkbox("AI Optimization")
AIBacktest = st.sidebar.checkbox("AI Backtest")

if All_Create:
    AIOpt = True
    AIBacktest = True
if AIBacktest:
    st.sidebar.write('<hr>', unsafe_allow_html=True)

    # Start day for Ticker
    col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
    with col_year_start:
        year_start = st.text_input("Year (Start)", value='2013')
    with col_month_start:
        month_start = st.text_input("Month (Start)", value='01')
    with col_day_start:
        day_start = st.text_input("Day (Start)", value='01')
    Startday = f"{year_start}-{month_start}-{day_start}"
    
    current_date = datetime.now()
    # End day for Ticker
    col_year_end, col_month_end, col_day_end = st.sidebar.columns([1, 1, 1])
    with col_year_end:
        year_end = st.text_input("Year (End)", value=str(current_date.year))
    with col_month_end:
        month_end = st.text_input("Month (End)", value=str(current_date.month).zfill(2))
    with col_day_end:
        day_end = st.text_input("Day (End)", value=str(current_date.day).zfill(2))
    Endday = f"{year_end}-{month_end}-{day_end}"

    # Lend day
    col_year_lend, col_month_lend, col_day_lend = st.sidebar.columns([1, 1, 1])
    with col_year_lend:
        year_lend = st.text_input("Year (Lend)", value='2022')
    with col_month_lend:
        month_lend = st.text_input("Month (Lend)", value='10')
    with col_day_lend:
        day_lend = st.text_input("Day (Lend)", value='31')
    Learnfinish = f"{year_lend}-{month_lend}-{day_lend}"

    # Tstart day
    col_year_tstart, col_month_tstart, col_day_tstart = st.sidebar.columns([1, 1, 1])
    with col_year_tstart:
        year_tstart = st.text_input("Year (Tstart)", value='2022')
    with col_month_tstart:
        month_tstart = st.text_input("Month (Tstart)", value='11')
    with col_day_tstart:
        day_tstart = st.text_input("Day (Tstart)", value='01')
    TestInitiate = f"{year_tstart}-{month_tstart}-{day_tstart}"

    st.sidebar.write('<hr>', unsafe_allow_html=True)

    Ticker = st.sidebar.text_input("Ticker input", value='SOXL')
    Days = st.sidebar.text_input("Estimate days", value=15)
    Max = st.sidebar.text_input("Estimate rate up (%)", value=15)
    
    Max_depth = st.sidebar.text_input("Max depth", value=7)
    Min_Child_Weight = st.sidebar.text_input("Min Child Weight", value=7)
    ETA = st.sidebar.text_input("eta", value=0.9105359765728025)
    SubSample = st.sidebar.text_input("Sub Sample", value=0.5977364190287533)
    ColSample = st.sidebar.text_input("Col Sample", value=0.5532080821871641)
    Alpha = st.sidebar.text_input("Alpha", value=0.014746924081661839)
    Lambda = st.sidebar.text_input("Lambda", value=0.31102214081307195)
    Gamma = st.sidebar.text_input("Gamma", value=0.8986914184781452)
    
    submitted = st.sidebar.button("Run")
    st.sidebar.write('<hr>', unsafe_allow_html=True)

    
    if submitted: 
        startday = Startday
        endday = Endday
        Learnend = Learnfinish
        Teststart = TestInitiate
        CODE = Ticker
        day = int(Days)
        MAX_th = float(Max)/100+1
        #説明変数の取得
        fred_lst = ['DGS10', 'VIXCLS'] #'NIKKEI225','SP500','NASDAQCOM', 'UNRATE', 'DGS2', 'DGS3MO', 'DFF', 'CORESTICKM159SFRBATL']
        df_fred = data.DataReader(fred_lst,'fred',startday, endday).asfreq("D")
        df_fred = df_fred.fillna(method='ffill')

        # 株価の取得
        df_target = yf.download(CODE, start=startday, end=endday)
        df_ticker=df_target[df_target.index >= Teststart]
        df = pd.merge(df_target, df_fred, how='left', left_index=True, right_index=True)
        # df.to_excel("target1.xlsx")
        #####################################
        # VIX diff & 10DGS diff
        df['Diff VIX_1'] = df['VIXCLS'] - df['VIXCLS'].shift(1)
        df['Diff VIX_5'] = df['VIXCLS'] - df['VIXCLS'].shift(5)
        df['Diff VIX_10'] = df['VIXCLS'] - df['VIXCLS'].shift(10)
        df['Diff DGS10_1'] = df['DGS10'] - df['DGS10'].shift(1)
        df['Diff DGS10_5'] = df['DGS10'] - df['DGS10'].shift(5)
        df['Diff DGS10_10'] = df['DGS10'] - df['DGS10'].shift(10)
        # 不要なVIX DGS10の列を削除
        df.drop(['DGS10', 'VIXCLS'], axis=1, inplace=True)

        # Overlap studies
        # 1.BBの計算
        #ボリンジャーバンドを計算
        indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df['bb_middle'] = indicator_bb.bollinger_mavg()
        df['bb_up'] = indicator_bb.bollinger_hband()
        df['bb_down'] = indicator_bb.bollinger_lband()

        # 乖離率の計算と追加
        df['Diff_bb_up'] = (df['bb_up'] - df['Close']) / df['Close']
        # 乖離率の計算と追加
        df['Diff_bb_down'] = (df['Close'] - df['bb_down']) / df['Close']
        # 不要なVIX DGS10の列を削除
        df.drop(['bb_up', 'bb_down', 'bb_middle'], axis=1, inplace=True)

        # 2.SMAの計算
        df['5_SMA'] = df['Close'].rolling(window=5).mean()
        df['10_SMA'] = df['Close'].rolling(window=10).mean()
        df['20_SMA'] = df['Close'].rolling(window=20).mean()
        df['50_SMA'] = df['Close'].rolling(window=50).mean()
        df['100_SMA'] = df['Close'].rolling(window=100).mean()
        df['200_SMA'] = df['Close'].rolling(window=200).mean()

        # 乖離率の計算と追加
        df['5_SMA_Divergence'] = (df['5_SMA']-df['Close']) / df['Close']
        df['10_SMA_Divergence'] = (df['10_SMA']-df['Close']) / df['Close']
        df['20_SMA_Divergence'] = (df['20_SMA']-df['Close']) / df['Close']
        df['50_SMA_Divergence'] = (df['50_SMA']-df['Close']) / df['Close']
        df['100_SMA_Divergence'] = (df['100_SMA']-df['Close']) / df['Close']
        df['200_SMA_Divergence'] = (df['200_SMA']-df['Close']) / df['Close']

        # 不要なSMAの列を削除
        df.drop(['5_SMA', '10_SMA', '20_SMA', '50_SMA', '100_SMA', '200_SMA'], axis=1, inplace=True)

        # 3.EMAの計算
        df['5_EMA'] = df['Close'].ewm(span=5).mean()
        df['10_EMA'] = df['Close'].ewm(span=10).mean()
        df['20_EMA'] = df['Close'].ewm(span=20).mean()
        df['50_EMA'] = df['Close'].ewm(span=50).mean()
        df['100_EMA'] = df['Close'].ewm(span=100).mean()
        df['200_EMA'] = df['Close'].ewm(span=200).mean()

        # 乖離率の計算と追加
        df['5_EMA_Divergence'] = (df['5_EMA']-df['Close']) / df['Close']
        df['10_EMA_Divergence'] = (df['10_EMA']-df['Close']) / df['Close']
        df['20_EMA_Divergence'] = (df['20_EMA']-df['Close']) / df['Close']
        df['50_EMA_Divergence'] = (df['50_EMA']-df['Close']) / df['Close']
        df['100_EMA_Divergence'] = (df['100_EMA']-df['Close']) / df['Close']
        df['200_EMA_Divergence'] = (df['200_EMA']-df['Close']) / df['Close']
        # 不要なEMAの列を削除
        df.drop(['5_EMA', '10_EMA', '20_EMA', '50_EMA', '100_EMA', '200_EMA'], axis=1, inplace=True)

        # Momentum Indicator
        # MACDの計算
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df['macd'] = macd.iloc[:, 0]
        df['histgram'] = macd.iloc[:, 1]
        df['signal'] = macd.iloc[:, 2]

        # 5.RSIの計算
        df['rsi_values'] = ta.momentum.rsi(df['Close'], window=14)

        # 6.ストキャスティクスの計算
        df['highest_high'] = df['High'].rolling(14).max()
        df['lowest_low'] = df['Low'].rolling(14).min()
            
        df['slowK'] = (df['Close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']) * 100
        df['slowD'] = df['slowK'].rolling(1).mean().rolling(3).mean()
        df.drop(['highest_high', 'lowest_low'], axis=1, inplace=True)

        # 7.Money flow indexの計算
        df['Money_flow_index'] = tas.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window = 14)

        # 8. Williams' %Rの計算
        df['Williams_%R'] = tas.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp= 14)

        # 9. MOMの計算
        df['Momentum'] = ta.momentum.roc(df['Close'], window=14)

        # Volume Indicator
        # 10. OBVの計算
        df['OBV'] = ta.volume.obv(df['Close'], df['Volume'])
        df['OBV 30'] = df['OBV'].ewm(span=30).mean()
        df['OBV Diff'] = (df['OBV 30']-df['OBV']) / df['OBV']
        
        df.drop(['OBV', 'OBV 30'], axis=1, inplace=True)

        # Volatility Indicator
        # 11. ATRの計算
        df['ATR'] = tas.volatility.average_true_range(df['High'], df['Low'], df['Close'],  window= 14)

        # Pattern recognition
        # 12.Hammer
        # df['Hammer'] = ta.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])

        # 13.Shootingstar
        # df['Shootingstar'] = ta.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        # 14.CDLHARAMI
        # df['CDLHARAMI'] = ta.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])

        # 15.CDLHARAMICROSS
        # df['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])

        # 16.CDLDOJI
        # df['CDLDOJI'] = ta.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])

        # 17.CDLENGULFING
        # df['CDLENGULFING'] = ta.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])

        # 18.CDLPIERCING
        # df['CDLPIERCING'] = ta.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])

        # 19.CDLDARKCLOUDCOVER
        # df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])

        # 20.CDLMORNINGSTAR
        # df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        # 21.CDLEVENINGSTAR
        # df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        # 22.CDLINVERTEDHAMMER
        # df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])

        # 23.CDLHANGINGMAN
        # df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])

        # 24.CDL3WHITESOLDIERS
        # df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])

        # 25.CDL3BLACKCROWS
        # df['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])

        # 26.CDLSPINNINGTOP
        # df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])

        # 27.CDLMARUBOZU
        # df['CDLMARUBOZU'] = ta.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
        #####################################

        # 上がるときflag=1、上がらないときflag=0
        df['MEAN'] = df["Close"].rolling(day).mean().shift(-day)
        df['MEAN_per'] = df['MEAN'] / df["Close"]
        df['flag'] = np.where(df['MEAN_per'] > MAX_th, 1, 0)

        # 未来指標は削除
        df = df.drop(columns=['MEAN','MEAN_per'])
        # df.to_excel("Result1.xlsx")
        #####################################
        df = df.fillna(method='ffill')
        # df.to_excel("target.xlsx")
        df_train = df[:Learnend]
        df_test = df[Teststart:]


        X_train = df_train.drop(["flag"], axis=1)
        X_test = df_test.drop(["flag"], axis=1)

        y_train = df_train["flag"]
        y_test = df_test["flag"]

        # XGBoost用のデータセットに変換
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # XGBoost用のパラメータに変更
        params = {
                    'objective': "reg:squarederror", # 二値分類の場合はこちらに変更
                    # 'boosting_type': 'gbdt',
                    'max_depth': int(Max_depth),
                    'min_child_weight': int(Min_Child_Weight),
                    'eta': float(ETA),
                    'subsample': float(SubSample),
                    'colsample_bytree': float(ColSample),
                    'alpha': float(Alpha),
                    'lambda': float(Lambda),
                    'gamma': float(Gamma),
                }

        # 学習と評価
        evals = [ (dtrain, 'train'), (dtest, 'test')]
        gbm = xgb.train(params, dtrain, evals=evals)

        # 特徴量の重要度を取得
        importance = gbm.get_score(importance_type='gain')
        total_gain = sum(importance.values())
        importance_ratio = {feature: value / total_gain for feature, value in importance.items()}

        sorted_importance = sorted(importance_ratio.items(), key=lambda x: x[1], reverse=True)

        for feature, ratio in sorted_importance:
            print(f"Feature: {feature}, Importance Ratio: {ratio}")

        # 予測値と実測値の取得
        df_result = pd.DataFrame()
        df_result["Act"] = y_test
        # 予測値の取得
        df_result['Pred'] = gbm.predict(dtest)
        # 予測値がマイナスの場合は0に修正
        df_result['Pred'] = np.where(df_result['Pred'] <= 0, 0, df_result['Pred'])
        df_result['Pred'] = np.where(df_result['Pred'] >= 1.5, 1, df_result['Pred'])
        df_result['Pred_binary'] = np.round(df_result['Pred'])
        # df_result.to_excel("Result.xlsx")
        print(df_result)

        # 混合行列（コンフュージョンマトリクス）を表示する関数を定義
        def showConfusionMatrix(true,pred,pred_type):
            cm = confusion_matrix(true, pred, labels=[1, 0])
            labels = [1, 0]
            # データフレームに変換
            cm_labeled = pd.DataFrame(cm, columns=labels, index=labels)
            # 結果の表示
            print("◆混合行列（",pred_type,"）◆")
            print(cm_labeled)
        showConfusionMatrix(df_result['Act'], df_result['Pred_binary'],"最大株価")

        report = classification_report(df_result['Act'], df_result['Pred_binary'])
        print(report)
        # 正解率の計算
        accuracy = accuracy_score(df_result['Act'], df_result['Pred_binary'])
        precision = precision_score(df_result['Act'], df_result['Pred_binary'])
        recall = recall_score(df_result['Act'], df_result['Pred_binary'])
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

        def CalculationSignal():
            df_ticker.loc[:, 'Prediction'] = df_result.loc[:,'Pred_binary']
            df_ticker.loc[:, 'Buy_signal'] = 0
            for i in range(0, len(df_ticker)):
                if df_ticker.loc[df_ticker.index[i], 'Prediction'] == 1 and df_ticker.loc[df_ticker.index[i-1], 'Prediction'] == 1 and df_ticker.loc[df_ticker.index[i-2], 'Prediction'] == 1:
                    df_ticker.loc[df_ticker.index[i], 'Buy_signal'] = 1
            # df_ticker.to_excel("test.xlsx")
            
            return df_ticker

        class SignalStrategy(Strategy):
            # 戦略定義
            def init(self):
                self.buy_signal = self.I(lambda: CalculationSignal()['Buy_signal'])
                self.entry_price = None

            def next(self):
                if not self.position:
                    if self.buy_signal:
                        self.position.entry_price = self.data.Close[-1]
                        self.buy()
                elif self.position:
                    entry_close_price = self.position.entry_price
                    current_price = self.data.Close[-1]
                    if current_price >= entry_close_price * 1.3 or current_price <= entry_close_price * 0.85:
                        self.position.close()


        bt = Backtest(df_ticker, SignalStrategy, cash=1000000, commission=0, exclusive_orders=False)
        output=bt.run()
        print(output)
        bt.plot()
        print(output._strategy)
        
        ####################
        col1, col2, col3= st.columns([1,1,1])
        with col1:
            Actual_Pred_11 = ((df_result['Act'] == 1) & (df_result['Pred_binary'] == 1)).sum()
            Actual_Pred_10 = ((df_result['Act'] == 1) & (df_result['Pred_binary'] == 0)).sum()
            Actual_Pred_01 = ((df_result['Act'] == 0) & (df_result['Pred_binary'] == 1)).sum()
            Actual_Pred_00 = ((df_result['Act'] == 0) & (df_result['Pred_binary'] == 0)).sum()
            cm_df = {
                '/': ['/', 1,0],
                '1': [1, Actual_Pred_11, Actual_Pred_01],
                '0': [0, Actual_Pred_10, Actual_Pred_00]
            }
            # print(cm_df)
            st.write("◆混合行列（最大株価）◆")
            st.dataframe(cm_df) 
        with col2:
            st.write("Accuracy: ", f"<span>{accuracy}</span>", unsafe_allow_html=True)
            st.write("Precision: ", f"<span>{precision}</span>", unsafe_allow_html=True)
            st.write("Recall: ", f"<span>{recall}</span>", unsafe_allow_html=True)
        
        with col3:
            st.write("Importance ratio top 10")
            counter = 0
            for feature, ratio in sorted_importance:
                st.write(feature+":"+f"<span>{ratio}</span>", unsafe_allow_html=True)
                counter += 1
                if counter >= 10:
                    break
        
        data_col1, data_col2 = st.columns([2,4])
        with data_col1:
            st.write("Data Table - Act Pred")
            st.dataframe(df_result, height=500)
        with data_col2:
            st.write("Data Table - Ticker and Buy signal")
            st.dataframe(CalculationSignal(), height=500)
        
        ticker_col1, ticker_col2 = st.columns([2,4])
        with ticker_col1:
            stoutput = output.drop(output.tail(3).index)
            st.write("Backtest result table")
            st.write(stoutput)
        with ticker_col2:
            fig = go.Figure()

         #Tickerのローソク足チャートを追加
            fig.add_trace(go.Candlestick(x=df_target.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name=Ticker))

             # グラフのレイアウトを設定
            fig.update_layout(title='Total chart for ' + Ticker,
                            yaxis_title='Price',
                            xaxis_rangeslider_visible=False)

            # グラフを表示
            st.plotly_chart(fig)

        st.write(df)
        st.write(params)
        # fig = go.Figure(data=bt.plot())
        # fig.show()
        # グラフを表示
        # st.plotly_chart(fig)

if AIOpt:
    st.sidebar.write('<hr>', unsafe_allow_html=True)

    # Start day for Ticker
    Opt_col_year_start, Opt_col_month_start, Opt_col_day_start = st.sidebar.columns([1, 1, 1])
    with Opt_col_year_start:
        Opt_year_start = st.text_input("O_Y_Start", value='2013')
    with Opt_col_month_start:
        Opt_month_start = st.text_input("O_M_Start", value='01')
    with Opt_col_day_start:
        Opt_day_start = st.text_input("O_D_Start", value='01')
    Opt_Startday = f"{Opt_year_start}-{Opt_month_start}-{Opt_day_start}"
    
    Opt_current_date = datetime.now()
    # End day for Ticker
    Opt_col_year_end, Opt_col_month_end, Opt_col_day_end = st.sidebar.columns([1, 1, 1])
    with Opt_col_year_end:
        Opt_year_end = st.text_input("O_Y_End)", value=str(Opt_current_date.year))
    with Opt_col_month_end:
        Opt_month_end = st.text_input("O_M_End)", value=str(Opt_current_date.month).zfill(2))
    with Opt_col_day_end:
        Opt_day_end = st.text_input("O_D_End)", value=str(Opt_current_date.day).zfill(2))
    Opt_Endday = f"{Opt_year_end}-{Opt_month_end}-{Opt_day_end}"

    # Lend day
    Opt_col_year_lend, Opt_col_month_lend, Opt_col_day_lend = st.sidebar.columns([1, 1, 1])
    with Opt_col_year_lend:
        Opt_year_lend = st.text_input("O_Y_Lend", value='2022')
    with Opt_col_month_lend:
        Opt_month_lend = st.text_input("O_M_Lend", value='10')
    with Opt_col_day_lend:
        Opt_day_lend = st.text_input("O_D_Lend", value='31')
    Opt_Learnfinish = f"{Opt_year_lend}-{Opt_month_lend}-{Opt_day_lend}"

    # Tstart day
    Opt_col_year_tstart, Opt_col_month_tstart, Opt_col_day_tstart = st.sidebar.columns([1, 1, 1])
    with Opt_col_year_tstart:
        Opt_year_tstart = st.text_input("O_Y_Tstart", value='2022')
    with Opt_col_month_tstart:
        Opt_month_tstart = st.text_input("O_M_Tstart", value='11')
    with Opt_col_day_tstart:
        Opt_day_tstart = st.text_input("O_D_Tstart", value='01')
    Opt_TestInitiate = f"{Opt_year_tstart}-{Opt_month_tstart}-{Opt_day_tstart}"

    st.sidebar.write('<hr>', unsafe_allow_html=True)

    Opt_Ticker = st.sidebar.text_input("O_Ticker input", value='SOXL')
    Opt_Days = st.sidebar.text_input("O_Estimate days", value=15)
    Opt_Max = st.sidebar.text_input("O_Estimate rate up (%)", value=15)
    TrialNumber = st.sidebar.text_input("Trial number", value =2500)
    O_submitted = st.sidebar.button("Run Opt")
    st.sidebar.write('<hr>', unsafe_allow_html=True)

    
    if O_submitted: 
        
        startday = Opt_Startday
        endday = Opt_Endday
        Learnend = Opt_Learnfinish
        Teststart = Opt_TestInitiate
        CODE = Opt_Ticker
        day = int(Opt_Days)
        MAX_th = float(Opt_Max)/100+1
        trial=int(TrialNumber)

        #説明変数の取得
        fred_lst = ['DGS10', 'VIXCLS'] #'NIKKEI225','SP500','NASDAQCOM', 'UNRATE', 'DGS2', 'DGS3MO', 'DFF', 'CORESTICKM159SFRBATL']
        df_fred = data.DataReader(fred_lst,'fred',startday, endday).asfreq("D")
        df_fred = df_fred.fillna(method='ffill')

        # 株価の取得
        df_target = yf.download(CODE, start=startday, end=endday)
        df_ticker=df_target[df_target.index >= Teststart]
        df = pd.merge(df_target, df_fred, how='left', left_index=True, right_index=True)
        print(df)
        #####################################
        # VIX diff & 10DGS diff
        df['Diff VIX_1'] = df['VIXCLS'] - df['VIXCLS'].shift(1)
        df['Diff VIX_5'] = df['VIXCLS'] - df['VIXCLS'].shift(5)
        df['Diff VIX_10'] = df['VIXCLS'] - df['VIXCLS'].shift(10)
        df['Diff DGS10_1'] = df['DGS10'] - df['DGS10'].shift(1)
        df['Diff DGS10_5'] = df['DGS10'] - df['DGS10'].shift(5)
        df['Diff DGS10_10'] = df['DGS10'] - df['DGS10'].shift(10)
        # 不要なVIX DGS10の列を削除
        df.drop(['DGS10', 'VIXCLS'], axis=1, inplace=True)

        # Overlap studies
        # 1.BBの計算
        #ボリンジャーバンドを計算
        indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df['bb_middle'] = indicator_bb.bollinger_mavg()
        df['bb_up'] = indicator_bb.bollinger_hband()
        df['bb_down'] = indicator_bb.bollinger_lband()

        # 乖離率の計算と追加
        df['Diff_bb_up'] = (df['bb_up'] - df['Close']) / df['Close']
        # 乖離率の計算と追加
        df['Diff_bb_down'] = (df['Close'] - df['bb_down']) / df['Close']
        # 不要なVIX DGS10の列を削除
        df.drop(['bb_up', 'bb_down', 'bb_middle'], axis=1, inplace=True)

        # 2.SMAの計算
        df['5_SMA'] = df['Close'].rolling(window=5).mean()
        df['10_SMA'] = df['Close'].rolling(window=10).mean()
        df['20_SMA'] = df['Close'].rolling(window=20).mean()
        df['50_SMA'] = df['Close'].rolling(window=50).mean()
        df['100_SMA'] = df['Close'].rolling(window=100).mean()
        df['200_SMA'] = df['Close'].rolling(window=200).mean()

        # 乖離率の計算と追加
        df['5_SMA_Divergence'] = (df['5_SMA']-df['Close']) / df['Close']
        df['10_SMA_Divergence'] = (df['10_SMA']-df['Close']) / df['Close']
        df['20_SMA_Divergence'] = (df['20_SMA']-df['Close']) / df['Close']
        df['50_SMA_Divergence'] = (df['50_SMA']-df['Close']) / df['Close']
        df['100_SMA_Divergence'] = (df['100_SMA']-df['Close']) / df['Close']
        df['200_SMA_Divergence'] = (df['200_SMA']-df['Close']) / df['Close']

        # 不要なSMAの列を削除
        df.drop(['5_SMA', '10_SMA', '20_SMA', '50_SMA', '100_SMA', '200_SMA'], axis=1, inplace=True)

        # 3.EMAの計算
        df['5_EMA'] = df['Close'].ewm(span=5).mean()
        df['10_EMA'] = df['Close'].ewm(span=10).mean()
        df['20_EMA'] = df['Close'].ewm(span=20).mean()
        df['50_EMA'] = df['Close'].ewm(span=50).mean()
        df['100_EMA'] = df['Close'].ewm(span=100).mean()
        df['200_EMA'] = df['Close'].ewm(span=200).mean()

        # 乖離率の計算と追加
        df['5_EMA_Divergence'] = (df['5_EMA']-df['Close']) / df['Close']
        df['10_EMA_Divergence'] = (df['10_EMA']-df['Close']) / df['Close']
        df['20_EMA_Divergence'] = (df['20_EMA']-df['Close']) / df['Close']
        df['50_EMA_Divergence'] = (df['50_EMA']-df['Close']) / df['Close']
        df['100_EMA_Divergence'] = (df['100_EMA']-df['Close']) / df['Close']
        df['200_EMA_Divergence'] = (df['200_EMA']-df['Close']) / df['Close']
        # 不要なEMAの列を削除
        df.drop(['5_EMA', '10_EMA', '20_EMA', '50_EMA', '100_EMA', '200_EMA'], axis=1, inplace=True)

        # Momentum Indicator
        # MACDの計算
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        df['macd'] = macd.iloc[:, 0]
        df['histgram'] = macd.iloc[:, 1]
        df['signal'] = macd.iloc[:, 2]

        # 5.RSIの計算
        df['rsi_values'] = ta.momentum.rsi(df['Close'], window=14)

        # 6.ストキャスティクスの計算
        df['highest_high'] = df['High'].rolling(14).max()
        df['lowest_low'] = df['Low'].rolling(14).min()
            
        df['slowK'] = (df['Close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low']) * 100
        df['slowD'] = df['slowK'].rolling(1).mean().rolling(3).mean()
        df.drop(['highest_high', 'lowest_low'], axis=1, inplace=True)

        # 7.Money flow indexの計算
        df['Money_flow_index'] = tas.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window = 14)

        # 8. Williams' %Rの計算
        df['Williams_%R'] = tas.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp= 14)

        # 9. MOMの計算
        df['Momentum'] = ta.momentum.roc(df['Close'], window=14)

        # Volume Indicator
        # 10. OBVの計算
        df['OBV'] = ta.volume.obv(df['Close'], df['Volume'])
        df['OBV 30'] = df['OBV'].ewm(span=30).mean()
        df['OBV Diff'] = (df['OBV 30']-df['OBV']) / df['OBV']
        
        df.drop(['OBV', 'OBV 30'], axis=1, inplace=True)
        # Volatility Indicator
        # 11. ATRの計算
        df['ATR'] = tas.volatility.average_true_range(df['High'], df['Low'], df['Close'],  window= 14)


        # Pattern recognition
        # 12.Hammer
        # df['Hammer'] = ta.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])

        # 13.Shootingstar
        # df['Shootingstar'] = ta.CDLSHOOTINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        # 14.CDLHARAMI
        # df['CDLHARAMI'] = ta.CDLHARAMI(df['Open'], df['High'], df['Low'], df['Close'])

        # 15.CDLHARAMICROSS
        # df['CDLHARAMICROSS'] = ta.CDLHARAMICROSS(df['Open'], df['High'], df['Low'], df['Close'])

        # 16.CDLDOJI
        # df['CDLDOJI'] = ta.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])

        # 17.CDLENGULFING
        # df['CDLENGULFING'] = ta.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])

        # 18.CDLPIERCING
        # df['CDLPIERCING'] = ta.CDLPIERCING(df['Open'], df['High'], df['Low'], df['Close'])

        # 19.CDLDARKCLOUDCOVER
        # df['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df['Open'], df['High'], df['Low'], df['Close'])

        # 20.CDLMORNINGSTAR
        # df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        # 21.CDLEVENINGSTAR
        # df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])

        # 22.CDLINVERTEDHAMMER
        # df['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df['Open'], df['High'], df['Low'], df['Close'])

        # 23.CDLHANGINGMAN
        # df['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(df['Open'], df['High'], df['Low'], df['Close'])

        # 24.CDL3WHITESOLDIERS
        # df['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df['Open'], df['High'], df['Low'], df['Close'])

        # 25.CDL3BLACKCROWS
        # df['CDL3BLACKCROWS'] = ta.CDL3BLACKCROWS(df['Open'], df['High'], df['Low'], df['Close'])

        # 26.CDLSPINNINGTOP
        # df['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(df['Open'], df['High'], df['Low'], df['Close'])

        # 27.CDLMARUBOZU
        # df['CDLMARUBOZU'] = ta.CDLMARUBOZU(df['Open'], df['High'], df['Low'], df['Close'])
        #####################################
        #目的指標を決定


        # 上がるときflag=1、上がらないときflag=0
        df['MEAN'] = df["Close"].rolling(day).mean().shift(-day)
        df['MEAN_per'] = df['MEAN'] / df["Close"]
        df['flag'] = np.where(df['MEAN_per'] > MAX_th, 1, 0)

        # 未来指標は削除
        df = df.drop(columns=['MEAN','MEAN_per'])
        df = df.fillna(method='ffill')

        df_train = df[:Learnend]
        df_test = df[Teststart:]
        X_train = df_train.drop(["flag"], axis=1)
        X_test = df_test.drop(["flag"], axis=1)

        y_train = df_train["flag"]
        y_test = df_test["flag"]
        # 目的関数の定義
        def objective(trial):
            # パラメータの提案
            params = {
                "objective": "reg:squarederror", # 目的関数を候補から選ぶ
                'max_depth': trial.suggest_int('max_depth', 3, 9),
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 9),
                'eta': trial.suggest_float('eta', 0.01, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                'alpha': trial.suggest_float('alpha', 0.01, 2.0, log=True),
                'lambda': trial.suggest_float('lambda', 0.01, 2.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.01, 2.0, log=True),
            }

            # データセットの準備
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            # 学習と評価
            evals = [(dtrain, "train"), (dtest, "test")]
            gbm = xgb.train(params, dtrain, evals=evals)

            df_result = pd.DataFrame()
            df_result["Act"] = y_test
            # 予測値の取得
            df_result['Pred'] = gbm.predict(dtest)
            # 予測値がマイナスの場合は0に修正
            df_result['Pred'] = np.where(df_result['Pred'] < 0, 0, df_result['Pred'])
            df_result['Pred'] = np.where(df_result['Pred'] >= 1.5, 1, df_result['Pred'])
            df_result['Pred_binary'] = np.round(df_result['Pred'])
            # precision = precision_score(df_result['Act'], df_result['Pred_binary']) 
            f1 = f1_score(df_result['Act'], df_result['Pred_binary'])  # averageを'micro', 'macro', 'weighted'またはNoneに変更
            # recall = recall_score(df_result['Act'], df_result['Pred_binary'])
            # roc_auc = roc_auc_score(df_result['Act'], df_result['Pred'])

            return f1

        # スタディオブジェクトの作成
        study = optuna.create_study(direction="maximize")

        # 目的関数の最適化
        study.optimize(objective, n_trials=trial)

        # 最適なパラメータと評価指標の確認
        print("Best params:", study.best_params)
        print("Best value:", study.best_value)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        # 最適なパラメータでモデルを再構築
        best_params = study.best_params
        best_gbm = xgb.train(best_params, dtrain)

        # 特徴量の重要度を取得
        importance = best_gbm.get_score(importance_type='gain')
        total_gain = sum(importance.values())
        importance_ratio = {feature: value / total_gain for feature, value in importance.items()}

        sorted_importance = sorted(importance_ratio.items(), key=lambda x: x[1], reverse=True)

        for feature, ratio in sorted_importance:
            print(f"Feature: {feature}, Importance Ratio: {ratio}")

        # テストデータに対して予測を行う
        df_result_best = pd.DataFrame()
        df_result_best["Act"] = y_test
        # 予測値の取得
        df_result_best['Pred'] = best_gbm.predict(dtest)
        # 予測値がマイナスの場合は0に修正
        df_result_best['Pred'] = np.where(df_result_best['Pred'] < 0, 0, df_result_best['Pred'])
        df_result_best['Pred'] = np.where(df_result_best['Pred'] >= 1.5, 1, df_result_best['Pred'])
        df_result_best['Pred_binary'] = np.round(df_result_best['Pred'])
        # df_result_best.to_excel("Result.xlsx")

        print(df_result_best)

        # 混合行列（コンフュージョンマトリクス）を表示する関数を定義
        def showConfusionMatrix(true,pred,pred_type):
            cm = confusion_matrix(true, pred, labels=[1, 0])
            labels = [1, 0]
            # データフレームに変換
            cm_labeled = pd.DataFrame(cm, columns=labels, index=labels)
            # 結果の表示
            print("◆混合行列（",pred_type,"）◆")
            print(cm_labeled)
        showConfusionMatrix(df_result_best['Act'], df_result_best['Pred_binary'],"最大株価")

        report = classification_report(df_result_best['Act'], df_result_best['Pred_binary'])
        print(report)
        # 正解率の計算
        accuracy = accuracy_score(df_result_best['Act'], df_result_best['Pred_binary'])
        precision = precision_score(df_result_best['Act'], df_result_best['Pred_binary'])
        recall = recall_score(df_result_best['Act'], df_result_best['Pred_binary'])
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)

        def CalculationSignal():
            df_ticker.loc[:, 'Prediction'] = df_result_best.loc[:,'Pred_binary']
            df_ticker['iwhere']=None
            df_ticker.loc[:, 'Buy_signal'] = 0
            for i in range(0, len(df_ticker)):
                df_ticker.loc[df_ticker.index[i], 'iwhere']=i
                if df_ticker.loc[df_ticker.index[i], 'Prediction'] == 1 and df_ticker.loc[df_ticker.index[i-1], 'Prediction'] == 1 and df_ticker.loc[df_ticker.index[i-2], 'Prediction'] == 1:
                    df_ticker.loc[df_ticker.index[i], 'Buy_signal'] = 1
            # df_ticker.to_excel("test.xlsx")

            return df_ticker
        class SignalStrategy(Strategy):
            # 戦略定義
            def init(self):
                self.buy_signal = self.I(lambda: CalculationSignal()['Buy_signal'])
                self.entry_price = None

            def next(self):
                if not self.position:
                    if self.buy_signal:
                        self.position.entry_price = self.data.Close[-1]
                        self.buy()
                elif self.position:
                    entry_close_price = self.position.entry_price
                    current_price = self.data.Close[-1]
                    if current_price >= entry_close_price * 1.3 or current_price <= entry_close_price * 0.85:
                        self.position.close()


        bt = Backtest(df_ticker, SignalStrategy, cash=1000000, commission=0, exclusive_orders=False)
        output=bt.run()
        print(output)
        bt.plot()
        print(output._strategy)
        
        st.write("Best params:", study.best_params)
        st.write("Best value (f1):", study.best_value)
        
        col1, col2, col3= st.columns([1,1,1])
        with col1:
            Actual_Pred_11 = ((df_result_best['Act'] == 1) & (df_result_best['Pred_binary'] == 1)).sum()
            Actual_Pred_10 = ((df_result_best['Act'] == 1) & (df_result_best['Pred_binary'] == 0)).sum()
            Actual_Pred_01 = ((df_result_best['Act'] == 0) & (df_result_best['Pred_binary'] == 1)).sum()
            Actual_Pred_00 = ((df_result_best['Act'] == 0) & (df_result_best['Pred_binary'] == 0)).sum()
            cm_df = {
                '/': ['/', 1,0],
                '1': [1, Actual_Pred_11, Actual_Pred_01],
                '0': [0, Actual_Pred_10, Actual_Pred_00]
            }
            # print(cm_df)
            st.write("◆混合行列（最大株価）◆")
            st.dataframe(cm_df) 
        with col2:
            st.write("Accuracy: ", f"<span>{accuracy}</span>", unsafe_allow_html=True)
            st.write("Precision: ", f"<span>{precision}</span>", unsafe_allow_html=True)
            st.write("Recall: ", f"<span>{recall}</span>", unsafe_allow_html=True)
        
        with col3:
            st.write("Importance ratio top 10")
            counter = 0
            for feature, ratio in sorted_importance:
                st.write(feature+":"+f"<span>{ratio}</span>", unsafe_allow_html=True)
                counter += 1
                if counter >= 10:
                    break
        
        data_col1, data_col2 = st.columns([2,4])
        with data_col1:
            st.write("Data Table - Act Pred")
            st.dataframe(df_result_best, height=500)
        with data_col2:
            st.write("Data Table - Ticker and Buy signal")
            st.dataframe(CalculationSignal(), height=500)
        
        ticker_col1, ticker_col2 = st.columns([2,4])
        with ticker_col1:
            stoutput = output.drop(output.tail(3).index)
            st.write("Backtest result table")
            st.write(stoutput)
        with ticker_col2:
            fig = go.Figure()

         #Tickerのローソク足チャートを追加
            fig.add_trace(go.Candlestick(x=df_target.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name=Opt_Ticker))

             # グラフのレイアウトを設定
            fig.update_layout(title='Total chart for ' + Opt_Ticker,
                            yaxis_title='Price',
                            xaxis_rangeslider_visible=False)

            # グラフを表示
            st.plotly_chart(fig)

        st.write(df)
        st.write(best_params)