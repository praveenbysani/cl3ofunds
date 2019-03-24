import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import datetime
import pandas as pd

class MLStrategy(bt.Strategy):
    params =(('profit_thr',0.005),
                ('loss_thr',0.01),)
    def __init__(self):
        #get the ML prediction data
        self.dailypred= self.datas[1].open
        #use the opening price of next bar
        self.minuteopen = self.datas[0].open
        self.order_refs = list()
    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
    def notify_trade(self,trade):
        #if trade.isopen:
        #    self.log('{} Trade ref:{} has been opened with size:{}, price:{} and value:{}'.format(
        #                            self.datetime.time(0),trade.ref,trade.size,trade.price,trade.value))
        if trade.isclosed:
            self.log('Trade ref:{} has been closed with pnl:{}, new size:{}'.format(trade.ref,trade.pnl,trade.size))

    def notify_order(self,order):
        '''
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('Buy Order Executed at price:{}'.format(order.executed.price))
            if order.issell():
                self.log('Sell Order Executed at price:{}'.format(order.executed.price))
        elif order.status in [order.Canceled]:
            self.log('Order Execution has been cancelled:{}'.format(order.ref))
        elif order.status in [order.Rejected]:
            self.log('Order Execution has been rejected:{}'.format(order.ref))
        '''
        # once the order has been accepted, remove the order from the reference queue
        if not order.alive() and order.ref in self.order_refs:
            self.order_refs.remove(order.ref)

    def next(self):
        '''
        print('Minute Date:{}, Closing Price:{}; Daily Date:{} Daily Prediction:{}'.format(
                    self.datas[0].datetime.date(0),self.minuteclose[0],self.datas[1].datetime.date(0),self.dailypred[0]))
        print('{},{},{}'.format(type(self.feeddate[0]),self.preddate[0],self.preddate[-1]))
        '''

        # when the orders are pending, don't execute new ones
        if len(self.order_refs)==3:
            return
        # when the date has changed in the prediction feed indicating the new day has started
        if self.datas[1].datetime.date(-1) < self.datas[1].datetime.date(0):
            #verification of whether the feed and prediction data are aligned
            if self.datas[1].datetime.date(0) == self.datas[0].datetime.date(0):
                #when the prediction is positive
                if self.dailypred[0] > 0:
                    #verify if its the first order of the day
                    if(self.datetime.time(0).strftime('%H%M')=='0901'):
                        #verify if the system is not already in position
                        if not self.position:
                            exec_price = self.minuteopen[0]
                            limit_price = self.minuteopen[0]*(1+self.params.profit_thr)
                            stop_price = self.minuteopen[0]*(1-self.params.loss_thr)
                            brackets = self.buy_bracket(limitprice=limit_price, price=exec_price, stopprice=stop_price)
                            self.order_refs = [bracket.ref for bracket in brackets]

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.feedopen = self.datas[1].open
        #use the date object from the prediction frame
        #self.predopen = self.datas[1].open
    def next(self):
        if(self.datetime.time(0).strftime('%H%M')=='0901'):
            #print('{}:{};{}'.format(self.datas[1].datetime.date(-1),self.datas[1].datetime.date(0),self.datas[0].datetime.date(0)))
            print('{} '.format(self.feedopen[0]))

if __name__=='__main__':
    data= btfeeds.GenericCSVData(
                dataname='../data/dowj_data_2017_2018.csv',
                fromdate=datetime.datetime(2017,1,1),
                todate=datetime.datetime(2019,1,1),
                #important to adjust the time frame to Minutes from the default Daily
                timeframe=bt.TimeFrame.Minutes,
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
                dataname='../data/dowj_2017_2018_preds.csv',
                fromdate=datetime.datetime(2017,1,1),
                todate=datetime.datetime(2019,1,1),
                #important to adjust the time frame to Minutes from the default Daily
                timeframe=bt.TimeFrame.Minutes,
                dtformat=('%Y-%m-%d'),
                datetime=0,
                time=-1,
                open=1,
                high=2,
                low=-1,
                close=-1,
                volume=-1,
                openinterest=-1)
    ## disable plotting of the ML predictions
    daily_data.plotinfo.plot=False
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(50000)
    cerebro.adddata(data)
    cerebro.adddata(daily_data)
    cerebro.addstrategy(MLStrategy)
    #cerebro.addstrategy(TestStrategy)
    cerebro.addsizer(bt.sizers.FixedSize,stake=1)
    #add different analyzers
    #cerebro.addanalyzer(btanalyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(btanalyzers.DrawDown,fund=False,_name='drawdown')
    #cerebro.addanalyzer(btanalyzers.TradeAnalyzer,_name='trade_analyzer')
    #cerebro.addanalyzer(btanalyzers.PeriodStats,_name='period_stats')
    strats_run=cerebro.run()
    ml_strat =strats_run[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    #print(ml_strat.analyzers.trade_analyzer.print())
    print(ml_strat.analyzers.drawdown.print())
    #print(ml_strat.analyzers.sharpe.print())
    #print(ml_strat.analyzers.period_stats.print())
    cerebro.plot(volume=False)
