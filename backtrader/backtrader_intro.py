import backtrader as bt
import backtrader.indicators as bind
import backtrader.feeds as bfeeds

import matplotlib
import os.path
import sys
import datetime
%matplotlib inline

class ExampleStrategy(bt.Strategy):
    params = (
        ('maperiod',15),
    )
    def __init__(self):
        self.dataclose= self.datas[0].close
        self.order=None
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0],period=self.params.maperiod)
    '''
    def notify_order(self,order):
        if order.status in [order.Submitted,order.Accepted]:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                print('Buy Executed @ price:{}'.format(order.executed.price))
            elif order.issell():
                print('Sell Executed @ price:{}'.format(order.executed.price))
        elif order.status in [order.Canceled,order.Margin,order.Rejected]:
                print('Order Cancelled')
        self.order=None
    '''
    def notify_trade(self,trade):
        if not trade.isclosed:
            return
        #print('Operation Profit: {0:.2f} Net Profit: {0:.2f}'.format(trade.pnl,trade.pnlcomm))
    def next(self):
        if self.order:
            return
        if not self.position:
            if self.dataclose[0] > self.sma[0]:
                print('BUY Created {time} :: price:{p}'.format(time=self.datas[0].datetime.date(0),p=self.dataclose[0]))
                self.order=self.buy()
                #self.bar_executed=len(self)
        else:
            if self.dataclose[0] < self.sma[0]:
                print('SELL Created {time} :: price:{p}'.format(time=self.datas[0].datetime.date(0),p=self.dataclose[0]))
                self.order = self.sell()

    def stop(self):
        print('MA period: {} , Portfolio Value: {}'.format(self.params.maperiod,self.broker.getvalue()))



class SmaCross(bt.SignalStrategy):
    params = (('pfast', 10), ('pslow', 30),)
    def __init__(self):
        sma1, sma2 = bt.ind.SMA(period=self.p.pfast), bt.ind.SMA(period=self.p.pslow)
        self.signal_add(bt.SIGNAL_LONG, bt.ind.CrossOver(sma1, sma2))


#modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
#datapath = os.path.join(modpath, '../../datas/orcl-1995-2014.txt')
data = bt.feeds.YahooFinanceData(
    dataname='MSFT',
    # Do not pass values before this date
    fromdate=datetime.datetime(2011, 1, 1),
    # Do not pass values after this date
    todate=datetime.datetime(2017, 12, 31),
    reverse=False)

cerebro = bt.Cerebro()
cerebro.adddata(data)
cerebro.addstrategy(ExampleStrategy)
#cerebro.optstrategy(ExampleStrategy,maperiod=range(10,20))
cerebro.broker.setcash(1000.0)
cerebro.addsizer(bt.sizers.FixedSize, stake=10)

#print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.run()

cerebro.broker.get_value()
#print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

cerebro.plot()
