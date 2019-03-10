import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import datetime


class TestStrategy(bt.Strategy):
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
        # Simply log the closing price of the series from the reference
        #self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:
                # BUY, BUY, BUY!!! (with default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.buyprice = self.dataclose[0]
                self.order = self.buy()

        else:
            # Already in the market ... we might sell
            if self.dataclose[0] > self.buyprice*(1.02):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE (TP), %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
            elif self.dataclose[0] < self.buyprice*(0.98):
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE (SL), %.2f' % self.dataclose[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()

data= btfeeds.GenericCSVData(
            dataname='../data/FCPO_2014_2018_0719.csv',
            fromdate=datetime.datetime(2017,1,1),
            todate=datetime.datetime(2018,1,1),
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
cerebro = bt.Cerebro()
cerebro.broker.setcash(10000)
cerebro.adddata(data)
cerebro.addstrategy(TestStrategy)
cerebro.addsizer(bt.sizers.FixedSize,stake=1)
cerebro.run()
#cerebro.plot()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
