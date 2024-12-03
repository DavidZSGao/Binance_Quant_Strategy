"""
Title: Backtrader Strategy V1
Author: Gao Zhesi
Created: December, 2024
Python Version: 3.10
Description: Grid trading strategy for Binance BTCUSDT with market regime detection.
             Features include:
             - Basic grid trading with dynamic position sizing
             - Market regime based on EMA crossovers and volatility
             - RSI-based entry/exit conditions
             - ATR-based stop loss and take profit
             - Real-time risk-free rate for Sharpe calculation
"""

import backtrader as bt
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from io import StringIO

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
        ('risk_per_trade', 0.02),
        ('grid_percentage', 0.01),
        ('num_grids', 10),
        ('rsi_period', 14),
        ('max_positions', 3),
        ('stop_loss_atr', 2.0),
        ('take_profit_atr', 3.0),
        ('rsi_oversold', 35),    # Less strict
        ('rsi_overbought', 65),  # Less strict
        ('trend_ema_period', 20),
    )
    
    def __init__(self):
        # Skip first 50 days to allow indicators to warm up
        self.warmup = 50
        
        # Technical indicators
        self.market_regime = MarketRegime()
        self.volume_profile = VolumeProfile()
        self.rsi = bt.indicators.RSI(period=self.p.rsi_period)
        self.atr = bt.indicators.ATR()
        self.ema = bt.indicators.EMA(period=self.p.trend_ema_period)
        
        # Track grid levels and positions
        self.grid_levels = []
        self.orders = {}
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0
        
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
            
    def next(self):
        if len(self) < self.warmup:  # Skip warmup period
            return
            
        # Get current market conditions
        current_price = self.data.close[0]
        regime = self.market_regime.regime[0]
        vol_strength = self.volume_profile.vol_strength[0]
        trend = 1 if current_price > self.ema[0] else -1
        
        # Adjust parameters based on market regime
        regime_adjustments = {
            0: {'grid_mult': 0.8, 'risk_mult': 1.2},   # UT_LV: Tighter grids, more risk
            1: {'grid_mult': 1.2, 'risk_mult': 0.8},   # UT_HV: Wider grids, less risk
            2: {'grid_mult': 0.8, 'risk_mult': 0.7},   # DT_LV: Tighter grids, much less risk
            3: {'grid_mult': 1.5, 'risk_mult': 0.5}    # DT_HV: Much wider grids, least risk
        }
        
        adj = regime_adjustments.get(regime, {'grid_mult': 1.0, 'risk_mult': 1.0})
        
        # Calculate grid levels with ATR
        grid_size = min(self.atr[0] / current_price, 0.1) * self.p.grid_percentage * adj['grid_mult']
        self.grid_levels = [
            current_price * (1 + j * grid_size)
            for j in range(-self.p.num_grids, self.p.num_grids + 1)
        ]
        
        # Manage existing positions first
        self.manage_positions(trend)
        
        # Place new orders if conditions are met
        if not self.position or abs(self.position.size) < self.p.max_positions:
            self.place_grid_orders(trend, adj['risk_mult'])
    
    def place_grid_orders(self, trend, risk_mult):
        # Cancel all pending orders first
        for order in self.broker.get_orders_open():
            self.broker.cancel(order)
            
        current_price = self.data.close[0]
        vol_strength = self.volume_profile.vol_strength[0]
        
        # Calculate position size with volume consideration
        cash = self.broker.get_cash()
        risk_amount = cash * self.p.risk_per_trade * risk_mult
        base_size = risk_amount / (self.atr[0] * 2)  # Use 2x ATR for risk
        position_size = base_size * min(vol_strength, 2.0)  # Scale with volume but cap at 2x
        position_size = min(position_size, cash / current_price * 0.95)  # Important safety cap
        
        # Sort grid levels by proximity to current price
        buy_levels = sorted([level for level in self.grid_levels if level < current_price], 
                          key=lambda x: abs(current_price - x))
        sell_levels = sorted([level for level in self.grid_levels if level > current_price],
                           key=lambda x: abs(current_price - x))
        
        # Place limited number of orders based on market conditions
        remaining_positions = self.p.max_positions - len(self.broker.get_orders_open())
        orders_placed = 0
        
        # Buy conditions with relaxed RSI
        for level in buy_levels[:remaining_positions]:
            if ((trend == 1 and self.rsi[0] < 45) or  # Less strict RSI in uptrend
                (trend == -1 and self.rsi[0] < 35)):  # Less strict RSI in downtrend
                self.buy(size=position_size, price=level, exectype=bt.Order.Limit)
                orders_placed += 1
                if orders_placed >= remaining_positions:
                    break
        
        # Sell conditions with relaxed RSI
        if self.position.size > 0:
            for level in sell_levels[:remaining_positions]:
                if ((trend == -1 and self.rsi[0] > 55) or  # Less strict RSI in downtrend
                    (trend == 1 and self.rsi[0] > 65)):    # Less strict RSI in uptrend
                    self.sell(size=min(position_size, self.position.size), 
                             price=level, exectype=bt.Order.Limit)
    
    def manage_positions(self, trend):
        if not self.position.size:
            return
            
        # Dynamic stop-loss based on trend
        stop_loss_mult = 2.0 if trend == 1 else 1.5  # Wider stop-loss in uptrend
        stop_price = self.position.price - (self.atr[0] * stop_loss_mult)
        
        # Take profit based on volume strength
        take_profit_mult = 3.0 if self.volume_profile.vol_strength[0] > 1.5 else 2.0
        take_profit_price = self.position.price + (self.atr[0] * take_profit_mult)
        
        # Execute stop-loss
        if self.data.close[0] <= stop_price:
            self.close()
            self.log(f'STOP LOSS - Price: {self.data.close[0]:.2f}')
            
        # Execute take-profit
        elif self.data.close[0] >= take_profit_price:
            self.close()
            self.log(f'TAKE PROFIT - Price: {self.data.close[0]:.2f}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
            else:
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
    
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
cerebro.broker.setcash(10000.0)  # Match initial cash from pure Python version
cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

# Add analyzers
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
