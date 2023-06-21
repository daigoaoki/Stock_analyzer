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
from ta.utils import dropna
from ta.volatility import BollingerBands
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from backtesting import Backtest, Strategy

import plotly.graph_objects as go

st.set_page_config(page_title = "Stock analyzing tool", layout="wide")
# st.markdown('''
# #<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
# #''', unsafe_allow_html=True)


# with open('style2.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

hide_st_style = """
            <style>
            footer {visibility: visible;}
            footer:after {content:'Copyright @ AI trader x'; display:block; position: relative;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


#st.sidebar.image("figure/MODECLOGO.png", width=100)
st.sidebar.subheader("Stock Analyzer Ver 0.0.1")

AI = st.sidebar.checkbox("AI")
Variable_MACD = st.sidebar.checkbox("Variable MACD")

if AI:
    # st.set_page_config(page_title = "STOCK ANALYZER", layout="wide")
    # hide_st_style = """
    #             <style>
    #             footer {visibility: visible;}
    #             footer:after {content:'Copyright @ AI Trader X'; display:block; position: relative;}
    #             </style>
    #             """
    # st.subheader("XGBoost & Backtesting & Optimiation") 
    # st.markdown(hide_st_style, unsafe_allow_html=True)
    st.sidebar.subheader("AI trading tool")


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
        Min_Child_Weight = st.sidebar.text_input("Min Child Weight", value=10)
        ETA = st.sidebar.text_input("eta", value=0.6455701967682373)
        SubSample = st.sidebar.text_input("Sub Sample", value=0.32553449232155385)
        ColSample = st.sidebar.text_input("Col Sample", value=0.6036929511441536)
        Alpha = st.sidebar.text_input("Alpha", value=0.05784074450141299)
        Lambda = st.sidebar.text_input("Lambda", value=1.1340577490452297)
        Gamma = st.sidebar.text_input("Gamma", value=0.6281967214548834)
        
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
                        'boosting_type': 'gbdt',
                        'max_depth': Max_depth,
                        'min_child_weight': Min_Child_Weight,
                        'eta': ETA,
                        'subsample': SubSample,
                        'colsample_bytree': ColSample,
                        'alpha': Alpha,
                        'lambda': Lambda,
                        'gamma': Gamma,
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
                    'max_depth': trial.suggest_int('max_depth', 1, 9),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'eta': trial.suggest_float('eta', 0.01, 1.0, log=True),
                    'subsample': trial.suggest_float('subsample', 0.0, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.0, 1.0),
                    'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                    'lambda': trial.suggest_float('lambda', 0.01, 10.0, log=True),
                    'gamma': trial.suggest_float('gamma', 0.01, 10.0, log=True),
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
if Variable_MACD:
    st.sidebar.subheader("Variable MACD")
    All_Create = st.sidebar.checkbox("All")
    Monthly_Variable_MACD = st.sidebar.checkbox("Monthly Variable MACD")
    Weekly_Variable_MACD = st.sidebar.checkbox("Weekly Variable MACD")
    Dayly_Variable_MACD = st.sidebar.checkbox("Dayly Variable MACD")

    if Monthly_Variable_MACD:
        st.sidebar.write('<hr>', unsafe_allow_html=True)

        # Start day for Ticker
        col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
        with col_year_start:
            year_start = st.text_input("Year (Start)", value='1990')
        with col_month_start:
            month_start = st.text_input("Month (Start)", value='01')
        with col_day_start:
            day_start = st.text_input("Day (Start)", value='01')
        Startday = f"{year_start}-{month_start}-{day_start}"
        
        Ticker = st.sidebar.text_input("Ticker input", value='^NDX')
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.8)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=2.1)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=24)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=66)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=11)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=17)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=20)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=2)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
        if submitted:
                
            # multiprocessing.set_start_method('fork')
            candle_input = int(Candle)
            counter_candle = candle_input+1

            #ターゲットを指定
            ticker = Ticker
            #データを収集
            dfm = yf.download(ticker, period='max', interval = "1mo")
            # 期間の編集（例：2020年以降のデータのみ抽出）
            dfm = dfm[dfm.index >= Startday]
            #データを収集
            dfd = yf.download(ticker, period='max', interval = "1d")
            # 期間の編集（例：2020年以降のデータのみ抽出）
            dfd = dfd[dfd.index >= Startday]
            # BBのBreakUpを確認
            def check_BBUp(df, i):
                if df.loc[:, ('High', i)] > df.loc[:, ('bb_bbh', i)]:
                    return True
                else:
                    return False
                
            # BBのBreakDownを確認    
            def check_BBDown(df, i):
                if df.loc[:, ('Low',i)] < df.loc[:, ('bb_bbl', i)]:
                    return True
                else:
                    return False

            # BBのBreakUPDownを確認    
            def check_BBUpDown(df, i):
                if (df['Low'][i] < df['bb_bbl'][i]) or (df['High'][i] > df['bb_bbh'][i]):
                    return True
                else:
                    return False

            # 早いmacdのGCの定義
            def check_FASTGC(df, i):
                if df['FASTmacd'][i] > df['FASTsignal'][i] and df['FASTmacd'][i-1] < df['FASTsignal'][i-1]:
                    return True
                else:
                    return False

            # 早いmacdのDCの定義
            def check_FASTDC(df, i):
                if df['FASTmacd'][i] < df['FASTsignal'][i] and df['FASTmacd'][i-1] > df['FASTsignal'][i-1]:
                    return True
                else:
                    return False

            # 早いmacdのGCDCの定義
            def check_FASTGCDC(df, i):
                if (df['FASTmacd'][i] < df['FASTsignal'][i] and df['FASTmacd'][i-1] > df['FASTsignal'][i-1]) or (df['FASTmacd'][i] > df['FASTsignal'][i] and df['FASTmacd'][i-1] < df['FASTsignal'][i-1]):
                    return True
                else:
                    return False
            # 遅いmacdのGCの定義
            def check_GC(df, i):
                if df['macd'][i] > df['signal'][i] and df['macd'][i-1] < df['signal'][i-1]:
                    return True
                else:
                    return False

            # 遅いmacdのDCの定義
            def check_DC(df, i):
                if df['macd'][i] < df['signal'][i] and df['macd'][i-1] > df['signal'][i-1]:
                    return True
                else:
                    return False

            # plotmacdのGCの定義
            def plot_check_GC(df, i):
                if df['plot_macd'][i] > df['plot_signal'][i] and df['plot_macd'][i-1] < df['plot_signal'][i-1]:
                    return True
                else:
                    return False

            # plotmacdのDCの定義
            def plot_check_DC(df, i):
                if df['plot_macd'][i] < df['plot_signal'][i] and df['plot_macd'][i-1] > df['plot_signal'][i-1]:
                    return True
                else:
                    return False

            # counter_candle間の古いFASTGCDCをチェック
            def oldest_gcdc_index(df,i):
                OldGCDC_index = i
                for j in range(i, (i-counter_candle-1), -1):
                    if check_FASTGCDC(df, j):
                        OldGCDC_index = j
                return OldGCDC_index

            # counter_candle間の新しいBBBreakをチェック
            def index_BBbreak(df,i):
                NewBBBreak_index = i-counter_candle
                for j in range((i-counter_candle), i+1):
                    if check_BBUpDown(df, j):
                        NewBBBreak_index = j
                return NewBBBreak_index

            def CalculationSignal(Wdef, Updef, Downdef, Sfast, Sslow, Ssignal, Ffast, Fslow, Fsignal):

                pd.set_option('display.max_columns', None)
                df=dfm
                
                #ボリンジャーバンドを計算
                indicator_bb_upper = BollingerBands(close=df["Close"], window=Wdef, window_dev=Updef)
                indicator_bb_lower = BollingerBands(close=df["Close"], window=Wdef, window_dev=Downdef)
                df['bb_bbm'] = indicator_bb_upper.bollinger_mavg()
                df['bb_bbh'] = indicator_bb_upper.bollinger_hband()
                df['bb_bbl'] = indicator_bb_lower.bollinger_lband()

                #MACDを計算
                macd = ta.macd(df["Close"], fast=Sfast, slow=Sslow, signal=Ssignal)
                df['macd'] = macd.iloc[:, 0]
                df['histgram'] = macd.iloc[:, 1]
                df['signal'] = macd.iloc[:, 2]

                #早いMACDを計算
                FAST_macd = ta.macd(df["Close"], fast=Ffast, slow=Fslow, signal=Fsignal)
                df['FASTmacd'] = FAST_macd.iloc[:, 0]
                df['FASThistgram'] = FAST_macd.iloc[:, 1]
                df['FASTsignal'] = FAST_macd.iloc[:, 2]


                # 初期値
                df['plot_macd'] = df['macd']
                df['plot_signal'] = df['signal']
                df['plot_histgram'] = df['histgram']
                

                ######################################
                
                ######################################
                df['buy_signal'] = None
                df['sell_signal'] = None
                df['buy_signal_index'] = 0
                df['sell_signal_index'] = 0
                df['OldGCDCindex'] = None
                df['OldGCDCindexcheck']= None
            #    df['iwhere']=None
            #    df['checkGCDC']=None
                df['BBBreakNew']=0
                df['BBcheck']=0
                current_buy_signal=0
                current_sell_signal=0

                buy_signal_index = 0
                sell_signal_index = 0
                #print(df)
                for i in range(1, len(df)):
                    OldGCDC_index = i
                    for j in range(i, (i-counter_candle-1),-1):
                        if check_FASTGCDC(df, j):
                            OldGCDC_index = j
                    df.loc[df.index[i], 'OldGCDCindex'] = OldGCDC_index
                    df.loc[df.index[i], 'OldGCDCindexcheck'] = oldest_gcdc_index(df,i)
                    
            #        df.loc[df.index[i], 'iwhere']=i
            #        if check_FASTGCDC(df, i):
            #            df.loc[df.index[i], 'checkGCDC']='GCDC'
                    
                    if df.loc[df.index[i-1],'buy_signal'] == 1:
                        current_buy_signal = i-1
                    df.loc[df.index[i], 'buy_signal_index'] = current_buy_signal
                    
                    if df.loc[df.index[i-1], 'sell_signal'] == 1:
                        current_sell_signal = i-1
                    df.loc[df.index[i], 'sell_signal_index'] = current_sell_signal
                    df.loc[df.index[i], 'BBBreakNew']=index_BBbreak(df,i)
                    if check_BBUpDown(df, i-3):
                        df.loc[df.index[i], 'BBcheck']=1
                    if df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']:
                        if check_BBUpDown(df, i):
                            df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
                            df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
                            df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
                        else:
                            df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'macd']
                            df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'signal']
                            df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'histgram']
                    
                    if df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'FASTmacd']:
                        if (i-oldest_gcdc_index(df,i)) <= candle_input:
                            df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
                            df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
                            df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
                        if (i-oldest_gcdc_index(df,i)) == 0:
                            df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
                            df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
                            df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
                        if (i-oldest_gcdc_index(df,i)) == counter_candle:
                            if (i-oldest_gcdc_index(df,i)) > (i-index_BBbreak(df,i)):
                                df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
                                df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
                                df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']    
                            if (i-oldest_gcdc_index(df,i)) == (i-index_BBbreak(df,i)):
                                if check_BBUpDown(df,i-counter_candle):
                                    if not check_FASTGCDC(df,i-counter_candle):
                                        df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
                                        df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
                                        df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
                                    if check_FASTGCDC(df,(i-counter_candle)) and ((df['buy_signal_index'][i] < (i-counter_candle) and df['buy_signal_index'][i] > df['sell_signal_index'][i]) or (df['sell_signal_index'][i] < (i-counter_candle)) and df['buy_signal_index'][i] < df['sell_signal_index'][i]):
                                        df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
                                        df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
                                        df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
                                    else:
                                        df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'macd']
                                        df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'signal']
                                        df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'histgram']
                                if not check_BBUpDown(df,i-counter_candle):
                                    if df['buy_signal_index'][i] < df['OldGCDCindex'][i] or df['sell_signal_index'][i] < df['OldGCDCindex'][i]:
                                        df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
                                        df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
                                        df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
                                    if df['buy_signal_index'][i] == df['OldGCDCindex'][i] or df['sell_signal_index'][i] == df['OldGCDCindex'][i]:
                                        df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'macd']
                                        df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'signal']
                                        df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'histgram']

                    if (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'macd']):
                        if plot_check_GC(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i] < df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'buy_signal'] = 1
                        if plot_check_DC(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'sell_signal'] = 1
                    
                    if (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd']):
                        if check_FASTGC(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i]< df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'buy_signal'] = 1
                        if check_FASTDC(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'sell_signal'] = 1
                            
                    if (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'FASTmacd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd']):
                        if plot_check_GC(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i] < df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'buy_signal'] = 1
                        if plot_check_DC(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'sell_signal'] = 1
                            
                    if ((df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'FASTmacd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd']) or (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd'])):
                        if check_GC(df, i) and check_BBUpDown(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i]< df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'buy_signal'] = 1
                        if check_DC(df, i) and check_BBUpDown(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
                            df.loc[df.index[i], 'sell_signal'] = 1

                #print(df)
            #    df.to_excel('output.xlsx', index=True)
                #plotlyでグラフを作成
            #    fig = go.Figure()

                #NDXのローソク足チャートを追加
            #    fig.add_trace(go.Candlestick(x=df.index,
            #                    open=df['Open'],
            #                    high=df['High'],
            #                    low=df['Low'],
            #                    close=df['Close'],
            #                    name='NDX'))
                #ボリンジャーバンドの上限、中央、下限を追加
            #    fig.add_trace(go.Scatter(x=df.index, y=df['bb_bbh'], name='Upper BB', line=dict(color='orange', width=1)))
            #    fig.add_trace(go.Scatter(x=df.index, y=df['bb_bbm'], name='Middle BB', line=dict(color='orange', width=1)))
            #    fig.add_trace(go.Scatter(x=df.index, y=df['bb_bbl'], name='Lower BB', line=dict(color='orange', width=1)))

                #MACDとシグナル線を追加
            #    fig.add_trace(go.Scatter(x=df.index, y=df['plot_macd'], name='MACD', line=dict(color='blue', width=1)))
            #    fig.add_trace(go.Scatter(x=df.index, y=df['plot_signal'], name='MACD Signal', line=dict(color='red', width=1)))

                #ヒストグラムの色のリストを作成
            #    colors = ['red' if x < 0 else 'green' for x in df['plot_histgram']]

                #ヒストグラムを追加
            #    fig.add_trace(go.Bar(x=df.index, y=df['plot_histgram'], name='Histgram', marker_color=colors))

                #グラフのレイアウトを設定
            #    fig.update_layout(title='NDX with Bollinger Bands and MACD',
            #                    yaxis_title='Price',
            #                    xaxis_rangeslider_visible=False)
                
                # Add buy and sell signal annotations
            #    for i in range(1, len(df)):
            #        if df.loc[df.index[i-1], 'buy_signal'] == 1:
            #            fig.add_annotation(x=df.index[i-1], y=df['Close'][i-1], text='Buy', showarrow=True, arrowhead=2, arrowcolor='green')
            #        if df.loc[df.index[i-1], 'sell_signal'] == 1:
            #            fig.add_annotation(x=df.index[i-1], y=df['Close'][i-1], text='Sell', showarrow=True, arrowhead=2, arrowcolor='red')
                #fig.show()
                ##############################
                #データを収集
                SPXcalc = dfd
                # 売買シグナルの追加
                buy_dates = df[df['buy_signal'] == 1].index  # Buy_signalが1の日付のリスト
                sell_dates = df[df['sell_signal'] == 1].index  # Sell_signalが1の日付のリスト

                # 日足のデータフレームに売買シグナルを追加
                SPXcalc = SPXcalc.copy()
                SPXcalc.loc[:, 'Actual_buy_signal'] = None  # 初期値を0に設定
                SPXcalc.loc[:, 'Actual_sell_signal'] = None  # 初期値を0に設定


                for buy_date in buy_dates:
                    month = buy_date.month
                    year = buy_date.year
                    last_day_of_month = SPXcalc[(SPXcalc.index.month == month) & (SPXcalc.index.year == year)].index.max()
                    SPXcalc.loc[last_day_of_month, 'Actual_buy_signal'] = 1

                for sell_date in sell_dates:
                    month = sell_date.month
                    year = sell_date.year
                    last_day_of_month = SPXcalc[(SPXcalc.index.month == month) & (SPXcalc.index.year == year)].index.max()
                    SPXcalc.loc[last_day_of_month, 'Actual_sell_signal'] = 1

                ###################################################
                SPXcalc.loc[:, 'Actual_buy_signal'] = SPXcalc['Actual_buy_signal'].astype(bool)
                SPXcalc.loc[:, 'Actual_sell_signal'] = SPXcalc['Actual_sell_signal'].astype(bool)
                #SPXcalc.to_excel('output_Days.xlsx', index=True)
                return SPXcalc
            # 戦略定義
            class SignalStrategy(Strategy):
                Wdef = int(BBWindow)
                Updef = float(BBUpSigma)
                Downdef = float(BBDownSigma)
                Sfast = int(SlowMacdFast)
                Sslow = int(SlowMacdSlow)
                Ssignal = int(SlowMacdSmoothing)
                Ffast = int(FastMacdFast)
                Fslow = int(FastMacdSlow)
                Fsignal = int(FastMacdSmoothing)
                # 戦略定義
                def init(self):
                    self.buy_signal = self.I(lambda: CalculationSignal(self.Wdef, self.Updef, self.Downdef, self.Sfast, self.Sslow, self.Ssignal, self.Ffast, self.Fslow, self.Fsignal)['Actual_buy_signal'])
                    self.sell_signal = self.I(lambda: CalculationSignal(self.Wdef, self.Updef, self.Downdef, self.Sfast, self.Sslow, self.Ssignal, self.Ffast, self.Fslow, self.Fsignal)['Actual_sell_signal'])
                def next(self):
                    if self.buy_signal and not self.sell_signal:
                        # 購入条件を満たし、売却条件を満たさない場合の処理
                        self.buy()
                    elif not self.buy_signal and self.sell_signal:
                        # 売却条件を満たし、購入条件を満たさない場合の処理
                        self.position.close()

            bt = Backtest(dfd, SignalStrategy, cash=1000000, commission=0, exclusive_orders=False)
            output=bt.run()
            #output=bt.optimize(Wdef=range(16, 21, 3), Updef=[2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9], Downdef=[1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], Sfast=range(23,30,3), Sslow=range(27, 34,3), Ssignal=range(21, 28,3), Ffast=range(9,16,3), Fslow=range(23, 30,3), Fsignal=range(2,9,3), maximize = 'Equity Final [$]',
            #                   constraint=lambda p: p.Ssignal < p.Sfast < p.Sslow and p.Fsignal < p.Ffast < p.Fslow and p.Fsignal<p.Ssignal and p.Ffast<p.Sfast and p.Fslow<p.Sslow)
            print(output)
            bt.plot()
            print(output._strategy)
            # output = bt.optimize(candle_input=range(1, 4, 1), Fslow=range(10, 30, 1), Fsignal=range(10, 30, 1))
            #   Wdef = 20
            #   Updef = 2.6
            #   Downdef = 1.6
            #   candle_input = 2
            #   Sfast = 26
            #   Sslow = 30
            #   Ssignal = 24
            #   Ffast = 12
            #   Fslow = 26
            #   Fsignal = 5

            ##############################
            
            stoutput = output.drop(output.tail(3).index)
            st.write("Backtest result table")
            st.write(stoutput)


# st.header("MODEC Process - Web Application Tools")
# st.markdown("Hello there!")
# st.markdown("This web application is a software that is designed with useful tools for process engineers in this Ver 0.0.1. Since the web application is MODEC in-house tool, it is possible to upgrade anytime.")
# st.markdown("Whenever new functions are added, it will be versioned up and shared with all process members !!")
# st.markdown("If you have any request or idea, please move on to the **Request Form**.")
# st.markdown("You can find the source code in the [BuLiAn GitHub Repository](https://github.com/tdenzl/BuLiAn)")
# st.markdown("If you are interested in how this app was developed check out my [Medium article](https://tim-denzler.medium.com/is-bayern-m%C3%BCnchen-the-laziest-team-in-the-german-bundesliga-770cfbd989c7)")
# st.text('')
# st.markdown("👇  *Other URL Link*  ")
# st.markdown("[MODEC Intranet](http://home.modec.com/Pages/default.aspx)")
# st.markdown("[Techstreet Enterprise](https://subscriptions.techstreet.com/subscriptions/index)")
# st.markdown("[Capital Projects](https://projdocs.modec.com/cp/component/main)")
# st.markdown("[My Timesheet](https://mobile.modec.com/sap/bc/ui5_ui5/ui2/ushell/shells/abap/FioriLaunchpad.html#TimeEntry-create&/)")
# st.markdown("[Kissflow](https://modecusa.kissflow.com/view/login)")