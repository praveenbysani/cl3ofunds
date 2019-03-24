import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import datetime
import pandas as pd

class TestStrategy(bt.Strategy):
    params =(('profit_thr',0.02),
                ('loss_thr',0.01),)
    def __init__(self):
        #get the ML prediction data
        self.dailypred= self.datas[1].open
        #use the opening price of next bar
        self.minuteopen = self.datas[0].open
        self.feeddate = self.datas[0].datetime
        #use the date object from the prediction frame
        self.preddate = self.datas[1].datetime
        self.brackets = None
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    def notify_order(self,order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('Buy Order Executed at price:{}'.format(order.executed.price))
            if order.issell():
                self.log('Sell Order Executed at price:{}'.format(order.executed.price))
        elif order.status in [order.Rejected,order.Canceled]:
            self.log('Order Execution has failed..')

    def next(self):
        #print('Minute Date:{}, Closing Price:{}; Daily Date:{} Daily Prediction:{}'.format(
        #            self.datas[0].datetime.date(0),self.minuteclose[0],self.datas[1].datetime.date(0),self.dailypred[0]))
        #print('{},{},{}'.format(type(self.feeddate[0]),self.preddate[0],self.preddate[-1]))

        # when the date has changed in the prediction feed indicating the new day has started
        if self.preddate[0] > self.preddate[-1]:
            #verification of whether the feed and prediction data are aligned
            if self.preddate[0] == self.feeddate[0]:
                #when the prediction is positive
                if self.dailypred[0] > 0:
                    #verify if the system is not already in position
                    if not self.position:
                        exec_price = self.minuteopen[0]
                        limit_price = self.minuteopen[0]*(1+self.params.profit_thr)
                        stop_price = self.minuteopen[0]*(1-self.params.loss_thr)
                        brackets = self.buy_bracket(limitprice=limit_price, price=exec_price, stopprice=stop_price)

class SMAStrategy(bt.Strategy):
    params = (
        ('exitbars', 5),
    )
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # To keep track of pending orders and buy price/commission
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sma = btind.SimpleMovingAverage(self.data.close,period=30)
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None
    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
    def next(self):
        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return
        # Check if we are in the market
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.buyprice = self.dataclose[0]
                self.order = self.buy()
        else:
            # Already in the market ... we might sell
            if self.dataclose[0] > self.buyprice*(1.02):
                self.log('SELL CREATE (TP), %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
            elif self.dataclose[0] < self.buyprice*(0.98):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE (SL), %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

if __name__=='__main__':
    data= btfeeds.GenericCSVData(
                dataname='../data/FCPO_2007-2017_backadjusted.csv',
                fromdate=datetime.datetime(2016,1,1),
                todate=datetime.datetime(2017,4,28),
                dtformat=('%Y%m%d'),
                tmformat=('%H%M'),
                datetime=0,
                time=1,
                open=2,
                high=3,
                low=4,
                close=5,
                volume=6,
                openinterest=-1)
    daily_data = btfeeds.GenericCSVData(
                dataname='../data/fcpo_2016_2017_preds.csv',
                fromdate=datetime.datetime(2006,1,1),
                todate=datetime.datetime(2017,4,28),
                dtformat=('%Y-%m-%d'),
                datetime=0,
                time=-1,
                open=1,
                high=-1,
                low=-1,
                close=-1,
                volume=-1,
                openinterest=-1)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000)
    cerebro.adddata(data)
    cerebro.adddata(daily_data)
    cerebro.addstrategy(TestStrategy)
    cerebro.addsizer(bt.sizers.FixedSize,stake=1)
    cerebro.run()
    #cerebro.plot()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
