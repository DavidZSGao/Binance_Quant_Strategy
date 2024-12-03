"""
Title: Backtrader Strategy V2
Author: Gao Zhesi
Created: December, 2024
Python Version: 3.10
Description: Dynamic grid trading strategy for Binance BTCUSDT that adapts to market conditions.
             Features include:
             - Market regime detection (trend & volatility based)
             - Dynamic grid sizing based on market regime
             - Volume profile analysis
             - Adaptive risk management with ATR-based stops
"""

import backtrader as bt
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

def get_risk_free_rate():
    """
    Fetch the current 3-month Treasury Bill rate as risk-free rate
    Returns rate as a decimal (e.g., 0.0525 for 5.25%)
    """
    try:
        # Using the Treasury Direct API
        url = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2023/all?field_tdr_date_value=2023&type=daily_treasury_bill_rates&page&_format=csv"
        response = requests.get(url)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            # Get the most recent 3-month rate
            latest_rate = df.iloc[0]['3 Mo'] / 100  # Convert percentage to decimal
            return latest_rate
        else:
            return 0.0525  # Default to current approximate rate of 5.25% if fetch fails
    except:
        return 0.0525  # Default to current approximate rate of 5.25% if fetch fails

# Fetch daily candlestick data from Binance API
BASE_URL = "https://data-api.binance.vision/api/v3/klines"
params = {
    "symbol": "BTCUSDT",   # Trading pair
    "interval": "1d",      # Daily candles
    "limit": 1000          # Max rows per request
}

response = requests.get(BASE_URL, params=params)

if response.status_code == 200:
    # Convert JSON to DataFrame
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "Open time", "Open", "High", "Low", "Close", "Volume",
        "Close time", "Quote asset volume", "Number of trades",
        "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
    ])
    
    # Convert timestamps to datetime
    df["Open time"] = pd.to_datetime(df["Open time"], unit='ms')
    df.set_index("Open time", inplace=True)
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
else:
    print(f"Error {response.status_code}: {response.text}")

class MarketRegime(bt.Indicator):
    lines = ('regime',)
    params = (('fast_ema', 20), ('slow_ema', 50), ('vol_period', 20))
    
    def __init__(self):
        self.fast_ema = bt.indicators.EMA(period=self.p.fast_ema)
        self.slow_ema = bt.indicators.EMA(period=self.p.slow_ema)
        self.volatility = bt.indicators.StdDev(self.data.close, period=self.p.vol_period)
        self.vol_ma = bt.indicators.SMA(self.volatility, period=self.p.vol_period)
        
    def next(self):
        trend = 1 if self.fast_ema[0] > self.slow_ema[0] else -1
        high_vol = self.volatility[0] > self.vol_ma[0]
        
        if trend == 1:  # Uptrend
            if high_vol:
                self.lines.regime[0] = 1  # UT_HV
            else:
                self.lines.regime[0] = 0  # UT_LV
        else:  # Downtrend
            if high_vol:
                self.lines.regime[0] = 3  # DT_HV
            else:
                self.lines.regime[0] = 2  # DT_LV

class VolumeProfile(bt.Indicator):
    lines = ('vol_strength',)
    params = (('period', 20),)
    
    def __init__(self):
        self.vol_ma = bt.indicators.SMA(self.data.volume, period=self.p.period)
        
    def next(self):
        self.lines.vol_strength[0] = self.data.volume[0] / self.vol_ma[0]

class EnhancedGridStrategy(bt.Strategy):
    params = (
        ('risk_per_trade', 0.02),     # Back to original 2%
        ('grid_percentage', 0.01),     # Back to 1%
        ('num_grids', 6),             # Fewer grids for more focused trading
        ('rsi_period', 14),
        ('max_positions', 3),
        ('stop_loss_atr', 2.0),
        ('take_profit_atr', 2.5),     # More conservative target
        ('rsi_oversold', 40),         # More lenient
        ('rsi_overbought', 60),       # More lenient
        ('trend_ema_period', 20),
        ('min_volume', 0.8)           # Minimum volume threshold
    )
    
    def __init__(self):
        self.warmup = 50
        
        # Technical indicators
        self.market_regime = MarketRegime()
        self.volume_profile = VolumeProfile()
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.atr = bt.indicators.ATR()
        self.ema = bt.indicators.EMA(period=self.p.trend_ema_period)
        self.ema20 = bt.indicators.EMA(period=20)
        self.ema50 = bt.indicators.EMA(period=50)
        
        # Track positions and performance
        self.grid_levels = []
        self.orders = {}
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        self.last_trade_exit = None
        
    def next(self):
        if len(self) < self.warmup:
            return
            
        current_price = self.data.close[0]
        regime = self.market_regime.regime[0]
        vol_strength = self.volume_profile.vol_strength[0]
        
        # Simplified trend detection
        trend = 1 if self.ema20[0] > self.ema50[0] else -1
        
        # Market regime adjustments
        regime_adjustments = {
            0: {'grid_mult': 0.9, 'risk_mult': 1.1},   # UT_LV
            1: {'grid_mult': 1.1, 'risk_mult': 0.9},   # UT_HV
            2: {'grid_mult': 0.8, 'risk_mult': 0.8},   # DT_LV
            3: {'grid_mult': 1.2, 'risk_mult': 0.6}    # DT_HV
        }
        
        adj = regime_adjustments.get(regime, {'grid_mult': 1.0, 'risk_mult': 1.0})
        
        # Simplified grid sizing
        grid_size = self.p.grid_percentage * adj['grid_mult']
        self.grid_levels = [
            current_price * (1 + j * grid_size)
            for j in range(-self.p.num_grids, self.p.num_grids + 1)
        ]
        
        self.manage_positions(trend)
        
        if self.last_trade_exit is None or len(self) - self.last_trade_exit > 2:
            if not self.position or abs(self.position.size) < self.p.max_positions:
                self.place_grid_orders(trend, adj['risk_mult'])
    
    def place_grid_orders(self, trend, risk_mult):
        for order in self.broker.get_orders_open():
            self.broker.cancel(order)
            
        current_price = self.data.close[0]
        vol_strength = self.volume_profile.vol_strength[0]
        
        if vol_strength < self.p.min_volume:
            return
        
        # Simplified position sizing
        cash = self.broker.get_cash()
        position_size = (cash * self.p.risk_per_trade * risk_mult) / current_price
        position_size = min(position_size, cash / current_price * 0.95)
        
        buy_levels = sorted([level for level in self.grid_levels if level < current_price], 
                          key=lambda x: abs(current_price - x))
        sell_levels = sorted([level for level in self.grid_levels if level > current_price],
                           key=lambda x: abs(current_price - x))
        
        remaining_positions = self.p.max_positions - len(self.broker.get_orders_open())
        
        # More lenient entry conditions
        for level in buy_levels[:remaining_positions]:
            if self.rsi[0] < self.p.rsi_oversold:
                self.buy(size=position_size/2, price=level, exectype=bt.Order.Limit)
        
        if self.position.size > 0:
            for level in sell_levels[:remaining_positions]:
                if self.rsi[0] > self.p.rsi_overbought:
                    size = min(position_size/2, self.position.size/2)
                    self.sell(size=size, price=level, exectype=bt.Order.Limit)
    
    def manage_positions(self, trend):
        if not self.position.size:
            return
            
        current_price = self.data.close[0]
        
        # Simplified stop-loss
        stop_loss_mult = self.p.stop_loss_atr * (1.2 if trend == 1 else 1.0)
        stop_price = self.position.price - (self.atr[0] * stop_loss_mult)
        
        # Simplified take-profit
        take_profit_mult = self.p.take_profit_atr
        take_profit_price = self.position.price + (self.atr[0] * take_profit_mult)
        
        if current_price < stop_price:
            self.log(f'STOP LOSS - Price: {current_price:.2f}')
            self.close()
        
        elif current_price > take_profit_price:
            self.log(f'TAKE PROFIT - Price: {current_price:.2f}')
            self.close()
            
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
    
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades += 1
            self.total_pnl += trade.pnl
            if trade.pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            # Log trade details
            self.log(f'TRADE CLOSED - PnL: {trade.pnl:.2f}, Total PnL: {self.total_pnl:.2f}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')

# Initialize Cerebro engine
cerebro = bt.Cerebro()
cerebro.addstrategy(EnhancedGridStrategy)

# Create a Data Feed
data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)

# Set initial cash and commission
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)

# Add analyzers with more detailed metrics
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=get_risk_free_rate(), annualize=True)
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')

# Run the backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
results = cerebro.run()
strat = results[0]

# Print performance metrics
print('\nPerformance Metrics:')
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Get trade analysis
trade_analysis = strat.analyzers.trades.get_analysis()
print('\nDetailed Trade Analysis:')
print(f'Total Trades: {trade_analysis.total.total}')
print(f'Won: {trade_analysis.won.total}')
print(f'Lost: {trade_analysis.lost.total}')
if trade_analysis.won.total > 0:
    print(f'Win Rate: {(trade_analysis.won.total/trade_analysis.total.total)*100:.2f}%')
    print(f'Average Win: {trade_analysis.won.pnl.average:.2f}')
    print(f'Max Win: {trade_analysis.won.pnl.max:.2f}')
if trade_analysis.lost.total > 0:
    print(f'Average Loss: {trade_analysis.lost.pnl.average:.2f}')
    print(f'Max Loss: {trade_analysis.lost.pnl.max:.2f}')

# Print other metrics
try:
    sharpe = strat.analyzers.sharpe.get_analysis()['sharperatio']
    print(f'\nSharpe Ratio: {sharpe:.2f}')
except:
    print('\nSharpe Ratio: N/A')

try:
    max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    print(f'Max Drawdown: {max_dd:.2f}%')
except:
    print('Max Drawdown: N/A')

try:
    sqn = strat.analyzers.sqn.get_analysis()['sqn']
    print(f'System Quality Number (SQN): {sqn:.2f}')
except:
    print('SQN: N/A')

try:
    vwr = strat.analyzers.vwr.get_analysis()['vwr']
    print(f'Variability-Weighted Return: {vwr:.2f}%')
except:
    print('VWR: N/A')

try:
    initial_value = 10000.0  # Initial portfolio value
    final_value = cerebro.broker.getvalue()
    total_return = ((final_value - initial_value) / initial_value) * 100
    print(f'Total Return: {total_return:.2f}%')
except:
    print('Total Return: N/A')

# Plot the result
cerebro.plot(style='candlestick')
