import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
import pandas_ta as ta
import streamlit as st
from ta.utils import dropna
from ta.volatility import BollingerBands
# import openpyxl
from backtesting import Backtest, Strategy
# import multiprocessing

import plotly.graph_objects as go

st.set_page_config(page_title = "STOCK ANALYZER", layout="wide")
hide_st_style = """
            <style>
            footer {visibility: visible;}
            footer:after {content:'Copyright @ AI Trader X'; display:block; position: relative;}
            </style>
            """
st.subheader("Variable MACD") 
st.markdown(hide_st_style, unsafe_allow_html=True)
st.sidebar.subheader("STOCK ANALYZER TOOL V0.0.1")


All_Create = st.sidebar.checkbox("All")
Monthly_Variable_MACD = st.sidebar.checkbox("Monthly Variable MACD")
Optimization_Monthly_Variable_MACD = st.sidebar.checkbox("Optimization Monthly Variable MACD")
Weekly_Variable_MACD = st.sidebar.checkbox("Weekly Variable MACD")
Dayly_Variable_MACD = st.sidebar.checkbox("Dayly Variable MACD")

if Monthly_Variable_MACD:
    #Checkboxの作成
    Function_Type = st.sidebar.selectbox(
        '- Select "Analyze ticker"',
        ("---", "^NDX", "^SPX", "^SOX", "XLK (TECL original)", "XBI (LABU original)", "KWEB (CWEB original)", "TLT (TMF original)", "^NSEI", "^N225", "FDN (WEBL original)", "XLV (CURE original)", "Others")
    )
    if Function_Type == "^NDX":
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
    
    if Function_Type == "^SPX":
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
        
        Ticker = st.sidebar.text_input("Ticker input", value='^SPX')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.6)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=1.6)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=26)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=30)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=24)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=12)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=26)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=5)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
         
    if Function_Type == "^SOX":
        st.sidebar.write('<hr>', unsafe_allow_html=True)

        # Start day for Ticker
        col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
        with col_year_start:
            year_start = st.text_input("Year (Start)", value='2005')
        with col_month_start:
            month_start = st.text_input("Month (Start)", value='01')
        with col_day_start:
            day_start = st.text_input("Day (Start)", value='01')
        Startday = f"{year_start}-{month_start}-{day_start}"
        
        Ticker = st.sidebar.text_input("Ticker input", value='^SOX')
        
        Candle = st.sidebar.text_input("Continue candle input", value=3)
        BBWindow = st.sidebar.text_input("BB window", value=25)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.3)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=1.3)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=53)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=80)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=35)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=4)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=16)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=2)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
    if Function_Type == "XLK (TECL original)":
        st.sidebar.write('<hr>', unsafe_allow_html=True)

        # Start day for Ticker
        col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
        with col_year_start:
            year_start = st.text_input("Year (Start)", value='2002')
        with col_month_start:
            month_start = st.text_input("Month (Start)", value='08')
        with col_day_start:
            day_start = st.text_input("Day (Start)", value='01')
        Startday = f"{year_start}-{month_start}-{day_start}"
        
        Ticker = st.sidebar.text_input("Ticker input", value='XLK')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=3.0)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=1.2)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=5)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=17)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=41)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=7)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=24)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=23)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
    
    if Function_Type == "XBI (LABU original)":
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
        
        Ticker = st.sidebar.text_input("Ticker input", value='XBI')
        
        Candle = st.sidebar.text_input("Continue candle input", value=5)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.9)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=1.9)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=22)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=47)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=30)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=6)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=15)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=5)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
    if Function_Type == "KWEB (CWEB original)":
        st.sidebar.write('<hr>', unsafe_allow_html=True)

        # Start day for Ticker
        col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
        with col_year_start:
            year_start = st.text_input("Year (Start)", value='2018')
        with col_month_start:
            month_start = st.text_input("Month (Start)", value='01')
        with col_day_start:
            day_start = st.text_input("Day (Start)", value='01')
        Startday = f"{year_start}-{month_start}-{day_start}"
        
        Ticker = st.sidebar.text_input("Ticker input", value='KWEB')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.4)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=1.6)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=22)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=47)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=27)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=8)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=18)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=6)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
    if Function_Type == "TLT (TMF original)":
        st.sidebar.write('<hr>', unsafe_allow_html=True)

        # Start day for Ticker
        col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
        with col_year_start:
            year_start = st.text_input("Year (Start)", value='2010')
        with col_month_start:
            month_start = st.text_input("Month (Start)", value='01')
        with col_day_start:
            day_start = st.text_input("Day (Start)", value='01')
        Startday = f"{year_start}-{month_start}-{day_start}"
        
        Ticker = st.sidebar.text_input("Ticker input", value='TLT')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.6)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=1.6)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=45)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=80)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=35)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=12)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=33)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=3)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
    if Function_Type == "^NSEI":
        st.sidebar.write('<hr>', unsafe_allow_html=True)

        # Start day for Ticker
        col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
        with col_year_start:
            year_start = st.text_input("Year (Start)", value='2011')
        with col_month_start:
            month_start = st.text_input("Month (Start)", value='01')
        with col_day_start:
            day_start = st.text_input("Day (Start)", value='01')
        Startday = f"{year_start}-{month_start}-{day_start}"
        
        Ticker = st.sidebar.text_input("Ticker input", value='^NSEI')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.5)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=2.2)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=22)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=47)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=22)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=4)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=10)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=4)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
    if Function_Type == "^N225":
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
        
        Ticker = st.sidebar.text_input("Ticker input", value='^N225')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.3)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=2.4)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=22)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=47)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=19)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=10)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=21)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=6)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
    if Function_Type == "FDN (WEBL original)":
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
        
        Ticker = st.sidebar.text_input("Ticker input", value='FDN')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.9)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=2.0)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=36)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=78)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=8)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=5)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=13)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=3)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)

    if Function_Type == "XLV (CURE original)":
        st.sidebar.write('<hr>', unsafe_allow_html=True)

        # Start day for Ticker
        col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
        with col_year_start:
            year_start = st.text_input("Year (Start)", value='2001')
        with col_month_start:
            month_start = st.text_input("Month (Start)", value='01')
        with col_day_start:
            day_start = st.text_input("Day (Start)", value='01')
        Startday = f"{year_start}-{month_start}-{day_start}"
        
        Ticker = st.sidebar.text_input("Ticker input", value='XLV')
        
        Candle = st.sidebar.text_input("Continue candle input", value=2)
        BBWindow = st.sidebar.text_input("BB window", value=20)
        BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.8)
        BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=1.4)
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=39)
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=50)
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=44)
        FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=8)
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=25)
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=14)
        
        submitted = st.sidebar.button("Run")
        st.sidebar.write('<hr>', unsafe_allow_html=True)
        
    if Function_Type == "Others":
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
        
        Ticker = st.sidebar.text_input("Ticker input")
        
        Candle = st.sidebar.text_input("Continue candle input")
        BBWindow = st.sidebar.text_input("BB window")
        BBUpSigma = st.sidebar.text_input("BB Up Sigma")
        BBDownSigma = st.sidebar.text_input("BB Down Sigma")
        SlowMacdFast= st.sidebar.text_input("Slow macd fast period")
        SlowMacdSlow= st.sidebar.text_input("Slow macd slow period")
        SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period")
        FastMacdFast= st.sidebar.text_input("Fast macd fast period")
        FastMacdSlow= st.sidebar.text_input("Fast macd slow period")
        FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period")
        
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
            # 期間の編集（例：2020年以降のデータのみ抽出）
            df = df[df.index >= Startday]
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
            # plotlyでグラフを作成
            # st.title(ticker + 'with Bollinger Bands and MACD')

            # NDXのローソク足チャートを作成
            candlestick = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker)

            # ボリンジャーバンドの上限、中央、下限を作成
            upper_bb = go.Scatter(x=df.index, y=df['bb_bbh'], name='Upper BB', line=dict(color='blue', width=1))
            middle_bb = go.Scatter(x=df.index, y=df['bb_bbm'], name='Middle BB', line=dict(color='orange', width=1))
            lower_bb = go.Scatter(x=df.index, y=df['bb_bbl'], name='Lower BB', line=dict(color='blue', width=1))

            # MACDとシグナル線を作成
            macd = go.Scatter(x=df.index, y=df['plot_macd'], name='MACD', line=dict(color='blue', width=1))
            macd_signal = go.Scatter(x=df.index, y=df['plot_signal'], name='MACD Signal', line=dict(color='red', width=1))

            # ヒストグラムの色のリストを作成
            colors = ['red' if x < 0 else 'green' for x in df['plot_histgram']]

            # ヒストグラムを作成
            histogram = go.Bar(x=df.index, y=df['plot_histgram'], name='Histgram', marker_color=colors)

            # BuyとSellの注釈を作成
            annotations = []
            for i in range(1, len(df)):
                if df.loc[df.index[i-1], 'buy_signal'] == 1:
                    annotations.append(dict(x=df.index[i-1], y=df['Close'][i-1], text='Buy', showarrow=True, arrowhead=2, arrowcolor='green'))
                if df.loc[df.index[i-1], 'sell_signal'] == 1:
                    annotations.append(dict(x=df.index[i-1], y=df['Close'][i-1], text='Sell', showarrow=True, arrowhead=2, arrowcolor='red'))

            # グラフオブジェクトを作成
            fig = go.Figure(data=[candlestick, upper_bb, middle_bb, lower_bb, macd, macd_signal, histogram], layout=go.Layout(title=ticker + ' with Bollinger Bands and MACD', yaxis_title='Price', xaxis_rangeslider_visible=False, annotations=annotations))

            # グラフを表示
            # st.plotly_chart(fig)

            ##############################
            #データを収集
            df_days = dfd
            # 売買シグナルの追加
            buy_dates = df[df['buy_signal'] == 1].index  # Buy_signalが1の日付のリスト
            sell_dates = df[df['sell_signal'] == 1].index  # Sell_signalが1の日付のリスト

            # 日足のデータフレームに売買シグナルを追加
            df_days = df_days.copy()
            df_days.loc[:, 'Actual_buy_signal'] = None  # 初期値を0に設定
            df_days.loc[:, 'Actual_sell_signal'] = None  # 初期値を0に設定


            for buy_date in buy_dates:
                month = buy_date.month
                year = buy_date.year
                last_day_of_month = df_days[(df_days.index.month == month) & (df_days.index.year == year)].index.max()
                df_days.loc[last_day_of_month, 'Actual_buy_signal'] = 1

            for sell_date in sell_dates:
                month = sell_date.month
                year = sell_date.year
                last_day_of_month = df_days[(df_days.index.month == month) & (df_days.index.year == year)].index.max()
                df_days.loc[last_day_of_month, 'Actual_sell_signal'] = 1

            ###################################################
            df_days.loc[:, 'Actual_buy_signal'] = df_days['Actual_buy_signal'].astype(bool)
            df_days.loc[:, 'Actual_sell_signal'] = df_days['Actual_sell_signal'].astype(bool)
            #df_days.to_excel('output_Days.xlsx', index=True)
            return df_days, fig
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
                df_days, fig = CalculationSignal(self.Wdef, self.Updef, self.Downdef, self.Sfast, self.Sslow, self.Ssignal, self.Ffast, self.Fslow, self.Fsignal)
        
                self.buy_signal = self.I(lambda: df_days['Actual_buy_signal'])
                self.sell_signal = self.I(lambda: df_days['Actual_sell_signal'])
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
        Wdef = int(BBWindow)
        Updef = float(BBUpSigma)
        Downdef = float(BBDownSigma)
        Sfast = int(SlowMacdFast)
        Sslow = int(SlowMacdSlow)
        Ssignal = int(SlowMacdSmoothing)
        Ffast = int(FastMacdFast)
        Fslow = int(FastMacdSlow)
        Fsignal = int(FastMacdSmoothing)
        result = CalculationSignal(Wdef, Updef, Downdef, Sfast, Sslow, Ssignal, Ffast, Fslow, Fsignal)
        df_days = result[0]
        fig = result[1]
        
        st.plotly_chart(fig)
        stoutput = output.drop(output.tail(3).index)
        st.write("Backtest result table")
        st.write(stoutput)

# if Optimization_Monthly_Variable_MACD:
#     #Checkboxの作成
#     Function_Type = st.sidebar.selectbox(
#         '- Select "Analyze ticker"',
#         ("---", "^NDX", "^SPX", "^SOX", "XLK (TECL original)", "XBI (LABU original)", "KWEB (CWEB original)", "TLT (TMF original)", "^NSEI", "^N225", "FDN (WEBL original)", "XLV (CURE original)", "Others")
#     )
#     if Function_Type == "^NDX":
#         st.sidebar.write('<hr>', unsafe_allow_html=True)

#         # Start day for Ticker
#         col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
#         with col_year_start:
#             year_start = st.text_input("Year (Start)", value='1990')
#         with col_month_start:
#             month_start = st.text_input("Month (Start)", value='01')
#         with col_day_start:
#             day_start = st.text_input("Day (Start)", value='01')
#         Startday = f"{year_start}-{month_start}-{day_start}"
        
#         Ticker = st.sidebar.text_input("Ticker input", value='^NDX')
        
#         Candle = st.sidebar.text_input("Continue candle input", value=2)
#         BBWindow = st.sidebar.text_input("BB window", value=20)
#         BBUpSigma = st.sidebar.text_input("BB Up Sigma", value=2.8)
#         BBDownSigma = st.sidebar.text_input("BB Down Sigma", value=2.1)
#         SlowMacdFast= st.sidebar.text_input("Slow macd fast period", value=24)
#         SlowMacdSlow= st.sidebar.text_input("Slow macd slow period", value=66)
#         SlowMacdSmoothing= st.sidebar.text_input("Slow macd Smoothing period", value=11)
#         FastMacdFast= st.sidebar.text_input("Fast macd fast period", value=17)
#         FastMacdSlow= st.sidebar.text_input("Fast macd slow period", value=20)
#         FastMacdSmoothing= st.sidebar.text_input("Fast macd Smoothing period", value=2)
        
#         submitted = st.sidebar.button("Run")
#         st.sidebar.write('<hr>', unsafe_allow_html=True)
        
#     if submitted:
            
#         # multiprocessing.set_start_method('fork')
#         candle_input = int(Candle)
#         counter_candle = candle_input+1

#         #ターゲットを指定
#         ticker = Ticker
#         #データを収集
#         dfm = yf.download(ticker, period='max', interval = "1mo")
        
#         #データを収集
#         dfd = yf.download(ticker, period='max', interval = "1d")
#         # 期間の編集（例：2020年以降のデータのみ抽出）
#         dfd = dfd[dfd.index >= Startday]
#         # BBのBreakUpを確認
#         def check_BBUp(df, i):
#             if df.loc[:, ('High', i)] > df.loc[:, ('bb_bbh', i)]:
#                 return True
#             else:
#                 return False
            
#         # BBのBreakDownを確認    
#         def check_BBDown(df, i):
#             if df.loc[:, ('Low',i)] < df.loc[:, ('bb_bbl', i)]:
#                 return True
#             else:
#                 return False

#         # BBのBreakUPDownを確認    
#         def check_BBUpDown(df, i):
#             if (df['Low'][i] < df['bb_bbl'][i]) or (df['High'][i] > df['bb_bbh'][i]):
#                 return True
#             else:
#                 return False

#         # 早いmacdのGCの定義
#         def check_FASTGC(df, i):
#             if df['FASTmacd'][i] > df['FASTsignal'][i] and df['FASTmacd'][i-1] < df['FASTsignal'][i-1]:
#                 return True
#             else:
#                 return False

#         # 早いmacdのDCの定義
#         def check_FASTDC(df, i):
#             if df['FASTmacd'][i] < df['FASTsignal'][i] and df['FASTmacd'][i-1] > df['FASTsignal'][i-1]:
#                 return True
#             else:
#                 return False

#         # 早いmacdのGCDCの定義
#         def check_FASTGCDC(df, i):
#             if (df['FASTmacd'][i] < df['FASTsignal'][i] and df['FASTmacd'][i-1] > df['FASTsignal'][i-1]) or (df['FASTmacd'][i] > df['FASTsignal'][i] and df['FASTmacd'][i-1] < df['FASTsignal'][i-1]):
#                 return True
#             else:
#                 return False
#         # 遅いmacdのGCの定義
#         def check_GC(df, i):
#             if df['macd'][i] > df['signal'][i] and df['macd'][i-1] < df['signal'][i-1]:
#                 return True
#             else:
#                 return False

#         # 遅いmacdのDCの定義
#         def check_DC(df, i):
#             if df['macd'][i] < df['signal'][i] and df['macd'][i-1] > df['signal'][i-1]:
#                 return True
#             else:
#                 return False

#         # plotmacdのGCの定義
#         def plot_check_GC(df, i):
#             if df['plot_macd'][i] > df['plot_signal'][i] and df['plot_macd'][i-1] < df['plot_signal'][i-1]:
#                 return True
#             else:
#                 return False

#         # plotmacdのDCの定義
#         def plot_check_DC(df, i):
#             if df['plot_macd'][i] < df['plot_signal'][i] and df['plot_macd'][i-1] > df['plot_signal'][i-1]:
#                 return True
#             else:
#                 return False

#         # counter_candle間の古いFASTGCDCをチェック
#         def oldest_gcdc_index(df,i):
#             OldGCDC_index = i
#             for j in range(i, (i-counter_candle-1), -1):
#                 if check_FASTGCDC(df, j):
#                     OldGCDC_index = j
#             return OldGCDC_index

#         # counter_candle間の新しいBBBreakをチェック
#         def index_BBbreak(df,i):
#             NewBBBreak_index = i-counter_candle
#             for j in range((i-counter_candle), i+1):
#                 if check_BBUpDown(df, j):
#                     NewBBBreak_index = j
#             return NewBBBreak_index

#         def objective(trial):

#             pd.set_option('display.max_columns', None)
#             df=dfm
#             #ボリンジャーバンドを計算
#             indicator_bb_upper = BollingerBands(close=df["Close"], window=Wdef, window_dev=Updef)
#             indicator_bb_lower = BollingerBands(close=df["Close"], window=Wdef, window_dev=Downdef)
#             df['bb_bbm'] = indicator_bb_upper.bollinger_mavg()
#             df['bb_bbh'] = indicator_bb_upper.bollinger_hband()
#             df['bb_bbl'] = indicator_bb_lower.bollinger_lband()

#             #MACDを計算
#             macd = ta.macd(df["Close"], fast=Sfast, slow=Sslow, signal=Ssignal)
#             df['macd'] = macd.iloc[:, 0]
#             df['histgram'] = macd.iloc[:, 1]
#             df['signal'] = macd.iloc[:, 2]

#             #早いMACDを計算
#             FAST_macd = ta.macd(df["Close"], fast=Ffast, slow=Fslow, signal=Fsignal)
#             df['FASTmacd'] = FAST_macd.iloc[:, 0]
#             df['FASThistgram'] = FAST_macd.iloc[:, 1]
#             df['FASTsignal'] = FAST_macd.iloc[:, 2]


#             # 初期値
#             df['plot_macd'] = df['macd']
#             df['plot_signal'] = df['signal']
#             df['plot_histgram'] = df['histgram']
            

#             ######################################
#             # 期間の編集（例：2020年以降のデータのみ抽出）
#             df = df[df.index >= Startday]
#             ######################################
#             df['buy_signal'] = None
#             df['sell_signal'] = None
#             df['buy_signal_index'] = 0
#             df['sell_signal_index'] = 0
#             df['OldGCDCindex'] = None
#             df['OldGCDCindexcheck']= None
#         #    df['iwhere']=None
#         #    df['checkGCDC']=None
#             df['BBBreakNew']=0
#             df['BBcheck']=0
#             current_buy_signal=0
#             current_sell_signal=0

#             buy_signal_index = 0
#             sell_signal_index = 0
#             #print(df)
#             for i in range(1, len(df)):
#                 OldGCDC_index = i
#                 for j in range(i, (i-counter_candle-1),-1):
#                     if check_FASTGCDC(df, j):
#                         OldGCDC_index = j
#                 df.loc[df.index[i], 'OldGCDCindex'] = OldGCDC_index
#                 df.loc[df.index[i], 'OldGCDCindexcheck'] = oldest_gcdc_index(df,i)
                
#         #        df.loc[df.index[i], 'iwhere']=i
#         #        if check_FASTGCDC(df, i):
#         #            df.loc[df.index[i], 'checkGCDC']='GCDC'
                
#                 if df.loc[df.index[i-1],'buy_signal'] == 1:
#                     current_buy_signal = i-1
#                 df.loc[df.index[i], 'buy_signal_index'] = current_buy_signal
                
#                 if df.loc[df.index[i-1], 'sell_signal'] == 1:
#                     current_sell_signal = i-1
#                 df.loc[df.index[i], 'sell_signal_index'] = current_sell_signal
#                 df.loc[df.index[i], 'BBBreakNew']=index_BBbreak(df,i)
#                 if check_BBUpDown(df, i-3):
#                     df.loc[df.index[i], 'BBcheck']=1
#                 if df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']:
#                     if check_BBUpDown(df, i):
#                         df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
#                         df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
#                         df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
#                     else:
#                         df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'macd']
#                         df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'signal']
#                         df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'histgram']
                
#                 if df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'FASTmacd']:
#                     if (i-oldest_gcdc_index(df,i)) <= candle_input:
#                         df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
#                         df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
#                         df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
#                     if (i-oldest_gcdc_index(df,i)) == 0:
#                         df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
#                         df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
#                         df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
#                     if (i-oldest_gcdc_index(df,i)) == counter_candle:
#                         if (i-oldest_gcdc_index(df,i)) > (i-index_BBbreak(df,i)):
#                             df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
#                             df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
#                             df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']    
#                         if (i-oldest_gcdc_index(df,i)) == (i-index_BBbreak(df,i)):
#                             if check_BBUpDown(df,i-counter_candle):
#                                 if not check_FASTGCDC(df,i-counter_candle):
#                                     df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
#                                     df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
#                                     df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
#                                 if check_FASTGCDC(df,(i-counter_candle)) and ((df['buy_signal_index'][i] < (i-counter_candle) and df['buy_signal_index'][i] > df['sell_signal_index'][i]) or (df['sell_signal_index'][i] < (i-counter_candle)) and df['buy_signal_index'][i] < df['sell_signal_index'][i]):
#                                     df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
#                                     df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
#                                     df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
#                                 else:
#                                     df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'macd']
#                                     df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'signal']
#                                     df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'histgram']
#                             if not check_BBUpDown(df,i-counter_candle):
#                                 if df['buy_signal_index'][i] < df['OldGCDCindex'][i] or df['sell_signal_index'][i] < df['OldGCDCindex'][i]:
#                                     df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'FASTmacd']
#                                     df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'FASTsignal']
#                                     df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'FASThistgram']
#                                 if df['buy_signal_index'][i] == df['OldGCDCindex'][i] or df['sell_signal_index'][i] == df['OldGCDCindex'][i]:
#                                     df.loc[df.index[i], 'plot_macd'] = df.loc[df.index[i], 'macd']
#                                     df.loc[df.index[i], 'plot_signal'] = df.loc[df.index[i], 'signal']
#                                     df.loc[df.index[i], 'plot_histgram'] = df.loc[df.index[i], 'histgram']

#                 if (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'macd']):
#                     if plot_check_GC(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i] < df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'buy_signal'] = 1
#                     if plot_check_DC(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'sell_signal'] = 1
                
#                 if (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd']):
#                     if check_FASTGC(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i]< df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'buy_signal'] = 1
#                     if check_FASTDC(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'sell_signal'] = 1
                        
#                 if (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'FASTmacd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd']):
#                     if plot_check_GC(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i] < df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'buy_signal'] = 1
#                     if plot_check_DC(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'sell_signal'] = 1
                        
#                 if ((df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'FASTmacd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd']) or (df.loc[df.index[i-1], 'plot_macd'] == df.loc[df.index[i-1], 'macd']) and (df.loc[df.index[i], 'plot_macd'] == df.loc[df.index[i], 'FASTmacd'])):
#                     if check_GC(df, i) and check_BBUpDown(df, i) and (df['buy_signal_index'][i] == 0 or df['buy_signal_index'][i]< df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'buy_signal'] = 1
#                     if check_DC(df, i) and check_BBUpDown(df, i) and (df['sell_signal_index'][i] == 0 or df['buy_signal_index'][i] > df['sell_signal_index'][i]):
#                         df.loc[df.index[i], 'sell_signal'] = 1

#             #print(df)
#         #    df.to_excel('output.xlsx', index=True)
#             # plotlyでグラフを作成
#             # st.title(ticker + 'with Bollinger Bands and MACD')

#             # NDXのローソク足チャートを作成
#             candlestick = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name=ticker)

#             # ボリンジャーバンドの上限、中央、下限を作成
#             upper_bb = go.Scatter(x=df.index, y=df['bb_bbh'], name='Upper BB', line=dict(color='blue', width=1))
#             middle_bb = go.Scatter(x=df.index, y=df['bb_bbm'], name='Middle BB', line=dict(color='orange', width=1))
#             lower_bb = go.Scatter(x=df.index, y=df['bb_bbl'], name='Lower BB', line=dict(color='blue', width=1))

#             # MACDとシグナル線を作成
#             macd = go.Scatter(x=df.index, y=df['plot_macd'], name='MACD', line=dict(color='blue', width=1))
#             macd_signal = go.Scatter(x=df.index, y=df['plot_signal'], name='MACD Signal', line=dict(color='red', width=1))

#             # ヒストグラムの色のリストを作成
#             colors = ['red' if x < 0 else 'green' for x in df['plot_histgram']]

#             # ヒストグラムを作成
#             histogram = go.Bar(x=df.index, y=df['plot_histgram'], name='Histgram', marker_color=colors)

#             # BuyとSellの注釈を作成
#             annotations = []
#             for i in range(1, len(df)):
#                 if df.loc[df.index[i-1], 'buy_signal'] == 1:
#                     annotations.append(dict(x=df.index[i-1], y=df['Close'][i-1], text='Buy', showarrow=True, arrowhead=2, arrowcolor='green'))
#                 if df.loc[df.index[i-1], 'sell_signal'] == 1:
#                     annotations.append(dict(x=df.index[i-1], y=df['Close'][i-1], text='Sell', showarrow=True, arrowhead=2, arrowcolor='red'))

#             # グラフオブジェクトを作成
#             fig = go.Figure(data=[candlestick, upper_bb, middle_bb, lower_bb, macd, macd_signal, histogram], layout=go.Layout(title=ticker + ' with Bollinger Bands and MACD', yaxis_title='Price', xaxis_rangeslider_visible=False, annotations=annotations))

#             # グラフを表示
#             # st.plotly_chart(fig)

#             ##############################
#             #データを収集
#             df_days = dfd
#             # 売買シグナルの追加
#             buy_dates = df[df['buy_signal'] == 1].index  # Buy_signalが1の日付のリスト
#             sell_dates = df[df['sell_signal'] == 1].index  # Sell_signalが1の日付のリスト

#             # 日足のデータフレームに売買シグナルを追加
#             df_days = df_days.copy()
#             df_days.loc[:, 'Actual_buy_signal'] = None  # 初期値を0に設定
#             df_days.loc[:, 'Actual_sell_signal'] = None  # 初期値を0に設定


#             for buy_date in buy_dates:
#                 month = buy_date.month
#                 year = buy_date.year
#                 last_day_of_month = df_days[(df_days.index.month == month) & (df_days.index.year == year)].index.max()
#                 df_days.loc[last_day_of_month, 'Actual_buy_signal'] = 1

#             for sell_date in sell_dates:
#                 month = sell_date.month
#                 year = sell_date.year
#                 last_day_of_month = df_days[(df_days.index.month == month) & (df_days.index.year == year)].index.max()
#                 df_days.loc[last_day_of_month, 'Actual_sell_signal'] = 1

#             ###################################################
#             df_days.loc[:, 'Actual_buy_signal'] = df_days['Actual_buy_signal'].astype(bool)
#             df_days.loc[:, 'Actual_sell_signal'] = df_days['Actual_sell_signal'].astype(bool)
#             #df_days.to_excel('output_Days.xlsx', index=True)
#             return df_days, fig
#         # 戦略定義
#         class SignalStrategy(Strategy):
#             Wdef = int(BBWindow)
#             Updef = float(BBUpSigma)
#             Downdef = float(BBDownSigma)
#             Sfast = int(SlowMacdFast)
#             Sslow = int(SlowMacdSlow)
#             Ssignal = int(SlowMacdSmoothing)
#             Ffast = int(FastMacdFast)
#             Fslow = int(FastMacdSlow)
#             Fsignal = int(FastMacdSmoothing)
#             # 戦略定義
#             def init(self):
#                 df_days, fig = CalculationSignal(self.Wdef, self.Updef, self.Downdef, self.Sfast, self.Sslow, self.Ssignal, self.Ffast, self.Fslow, self.Fsignal)
        
#                 self.buy_signal = self.I(lambda: df_days['Actual_buy_signal'])
#                 self.sell_signal = self.I(lambda: df_days['Actual_sell_signal'])
#             def next(self):
#                 if self.buy_signal and not self.sell_signal:
#                     # 購入条件を満たし、売却条件を満たさない場合の処理
#                     self.buy()
#                 elif not self.buy_signal and self.sell_signal:
#                     # 売却条件を満たし、購入条件を満たさない場合の処理
#                     self.position.close()

#         bt = Backtest(dfd, SignalStrategy, cash=1000000, commission=0, exclusive_orders=False)
#         output=bt.run()
#         #output=bt.optimize(Wdef=range(16, 21, 3), Updef=[2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9], Downdef=[1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], Sfast=range(23,30,3), Sslow=range(27, 34,3), Ssignal=range(21, 28,3), Ffast=range(9,16,3), Fslow=range(23, 30,3), Fsignal=range(2,9,3), maximize = 'Equity Final [$]',
#         #                   constraint=lambda p: p.Ssignal < p.Sfast < p.Sslow and p.Fsignal < p.Ffast < p.Fslow and p.Fsignal<p.Ssignal and p.Ffast<p.Sfast and p.Fslow<p.Sslow)
#         print(output)
#         bt.plot()
#         print(output._strategy)
#         # output = bt.optimize(candle_input=range(1, 4, 1), Fslow=range(10, 30, 1), Fsignal=range(10, 30, 1))
#         #   Wdef = 20
#         #   Updef = 2.6
#         #   Downdef = 1.6
#         #   candle_input = 2
#         #   Sfast = 26
#         #   Sslow = 30
#         #   Ssignal = 24
#         #   Ffast = 12
#         #   Fslow = 26
#         #   Fsignal = 5

#         ##############################
#         Wdef = int(BBWindow)
#         Updef = float(BBUpSigma)
#         Downdef = float(BBDownSigma)
#         Sfast = int(SlowMacdFast)
#         Sslow = int(SlowMacdSlow)
#         Ssignal = int(SlowMacdSmoothing)
#         Ffast = int(FastMacdFast)
#         Fslow = int(FastMacdSlow)
#         Fsignal = int(FastMacdSmoothing)
#         result = CalculationSignal(Wdef, Updef, Downdef, Sfast, Sslow, Ssignal, Ffast, Fslow, Fsignal)
#         df_days = result[0]
#         fig = result[1]
        
#         st.plotly_chart(fig)
#         stoutput = output.drop(output.tail(3).index)
#         st.write("Backtest result table")
#         st.write(stoutput)
    