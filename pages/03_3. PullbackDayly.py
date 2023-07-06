from datetime import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backtesting import Backtest, Strategy
import streamlit as st

st.set_page_config(page_title = "STOCK ANALYZER", layout="wide")
hide_st_style = """
            <style>
            footer {visibility: visible;}
            footer:after {content:'Copyright @ AI Trader X'; display:block; position: relative;}
            </style>
            """
st.subheader("PullBack Daily") 
st.markdown(hide_st_style, unsafe_allow_html=True)
st.sidebar.subheader("STOCK ANALYZER TOOL V0.0.1")

All_Create = st.sidebar.checkbox("All")
PullBack = st.sidebar.checkbox("PullBack Dayly")

if All_Create:
    PullBack = True
if PullBack:
    st.sidebar.write('<hr>', unsafe_allow_html=True)
    
    # Start day for Ticker
    col_year_start, col_month_start, col_day_start = st.sidebar.columns([1, 1, 1])
    with col_year_start:
        year_start = st.text_input("Year (Start)", value='1995')
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
    
    st.sidebar.write('<hr>', unsafe_allow_html=True)
    CODE = st.sidebar.text_input("Ticker input", value='SOXL')
    TICKER = st.sidebar.text_input("Base input", value='SPY')
    Exit = st.sidebar.text_input("Exit days", value=20)
    Protect = st.sidebar.text_input("Protect days", value=60)
    Skip = st.sidebar.text_input("Skip days", value=4)
    
    submitted = st.sidebar.button("Run")
    st.sidebar.write('<hr>', unsafe_allow_html=True)

    
    if submitted: 
        startday = Startday
        endday = datetime.today().strftime(Endday)
        Code = CODE
        Ticker = TICKER
        # 株価の取得
        df_target = yf.download(Code, start=startday, end=endday)
        df_ticker = yf.download(Ticker)
        df = pd.merge(df_target, df_ticker, how='left', left_index=True, right_index=True)
        df.rename(columns={'Open_x': 'Open'}, inplace=True)
        df.rename(columns={'High_x': 'High'}, inplace=True)
        df.rename(columns={'Low_x': 'Low'}, inplace=True)
        df.rename(columns={'Close_x': 'Close'}, inplace=True)
        df.rename(columns={'Volume_x': 'Volume'}, inplace=True)
        # 不要な列を削除
        df.drop(['Adj Close_x', 'Adj Close_y', "Volume_y"], axis=1, inplace=True)

        def CalculationSignal(ExitDay, ProtectDay, SkipCheck):
            
            # df['iwhere']=None
            df['buy_cond'] = 0
            df['sell_cond'] = 0
            df['in_trade'] = False
            df['skip_trade'] = False
            df["lowest_low"]=0
            df['buy_signal'] = 0
            df['sell_signal'] = 0
            exit_after_x_days = ExitDay
            df['entry_day'] = 0
            df['entry_price'] = 0
            for i in range(1, len(df)):
                # df.loc[df.index[i], 'iwhere']=i
                df.loc[df.index[i], "in_trade"]=df.loc[df.index[i-1], "in_trade"]
                df.loc[df.index[i],'entry_day']=df.loc[df.index[i-1],'entry_day']
                df.loc[df.index[i],'entry_price']=df.loc[df.index[i-1],'entry_price']
                
                if not df.loc[df.index[i-ProtectDay:i], 'Low_y'].empty:
                    df.loc[df.index[i],"lowest_low"] = min(df.loc[df.index[i-ProtectDay:i], 'Low_y'])
                for j in range(i-SkipCheck,i):
                    if df.loc[df.index[j+1],'Close_y'] < df.loc[df.index[j],"lowest_low"]:
                        df.loc[df.index[i],'skip_trade'] = True
                        break
                    
                if df.loc[df.index[i],'Close_y'] < df.loc[df.index[i-1],'Low_y'] and df.loc[df.index[i],'Close_y'] < df.loc[df.index[i-2],'Low_y'] and df.loc[df.index[i],'Close_y'] < df.loc[df.index[i],'Open_y']:
                    df.loc[df.index[i],'buy_cond']=1
                
                if df.loc[df.index[i],'Close_y'] > df.loc[df.index[i-1],'High_y'] and df.loc[df.index[i],'Close_y'] > df.loc[df.index[i-2],'High_y'] and df.loc[df.index[i],'Close_y'] > df.loc[df.index[i-3],'High_y'] or df.loc[df.index[i],'Close_y']<df.loc[df.index[i],"lowest_low"]:
                    df.loc[df.index[i],'sell_cond']=1
                
                if df.loc[df.index[i], "in_trade"]==True:
                    if (i - df.loc[df.index[i], "entry_day"] > 1) and (i - df.loc[df.index[i], "entry_day"] >= exit_after_x_days or df.loc[df.index[i],'sell_cond']==1):
                        # sell at the close
                        df.loc[df.index[i], "sell_signal"]=1
                        df.loc[df.index[i], "in_trade"] = False
                        
                if df.loc[df.index[i], "in_trade"]==False:
                    if df.loc[df.index[i],'buy_cond']==1 and df.loc[df.index[i], "in_trade"] == False and df.loc[df.index[i], "skip_trade"]==False:
                        df.loc[df.index[i],'buy_signal'] = 1
                        df.loc[df.index[i],'entry_price']=df.loc[df.index[i],"Close"]
                        df.loc[df.index[i],'entry_day'] = i
                        df.loc[df.index[i], "in_trade"]=True
            # print(df)
            # df.to_excel('pullbacktest.xlsx', index=True)

            # plotlyでグラフを作成
            fig = go.Figure()

            # NDXのローソク足チャートを追加
            fig.add_trace(go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name=Code))

            # グラフのレイアウトを設定
            fig.update_layout(title=Code + ' with Bollinger Bands and MACD',
                            yaxis_title='Price',
                            xaxis_rangeslider_visible=False)

            # Add buy and sell signal annotations
            for i in range(1, len(df)):
                if df.loc[df.index[i-1], 'buy_signal'] == 1:
                    fig.add_annotation(x=df.index[i], y=df['Open'][i], text='Buy', showarrow=True, arrowhead=2, arrowcolor='green')
                if df.loc[df.index[i-1], 'sell_signal'] == 1:
                    fig.add_annotation(x=df.index[i], y=df['Open'][i], text='Sell', showarrow=True, arrowhead=2, arrowcolor='red')
            # fig.show()

            return df, fig

        class SignalStrategy(Strategy):
            ExitDay = int(Exit)
            ProtectDay = int(Protect)
            SkipCheck = int(Skip)

            # 戦略定義
            def init(self):
                df, fig = CalculationSignal(self.ExitDay, self.ProtectDay, self.SkipCheck)
        
                self.buy_signal = self.I(lambda: df['buy_signal'])
                self.sell_signal = self.I(lambda: df['sell_signal'])
            def next(self):
                if self.buy_signal and not self.sell_signal:
                    # 購入条件を満たし、売却条件を満たさない場合の処理
                    self.buy()
                elif not self.buy_signal and self.sell_signal:
                    # 売却条件を満たし、購入条件を満たさない場合の処理
                    self.position.close()

        bt = Backtest(df, SignalStrategy, cash=1000000, commission=0, exclusive_orders=False)
        # output=bt.optimize(ExitDay=range(3, 20, 1), ProtectDay=range(45,46,1), SkipCheck=range(4, 5, 1), maximize = 'Equity Final [$]')
        output=bt.run()
        print(output)
        bt.plot()
        print(output._strategy)
        
        ExitDay = int(Exit)
        ProtectDay = int(Protect)
        SkipCheck = int(Skip)
        result = CalculationSignal(ExitDay, ProtectDay, SkipCheck)
        df = result[0]
        fig = result[1]
        
        st.plotly_chart(fig)
        stoutput = output.drop(output.tail(3).index)
        st.write("Backtest result table")
        st.write(stoutput)