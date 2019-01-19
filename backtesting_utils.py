import pandas as pd

def calculate_long_returns(x,profit_field,pred_field,alt_pred_field,commission_price,num_units,profit_thr=0.03,loss_thr=0.01):
    return_myr=0
    #when the prediction is zero, the model accumulates no returns
    if x[pred_field]==0:
        return_myr=0
    #when the model makes a prediction
    elif x[pred_field]==1:
        #when it is correct, sell it off when it reaches max x% profit zone
        #when the opposite site prediction is true, do nothing
        if x[alt_pred_field]==1:
            return_myr=0
        elif x[profit_field]==1:
            return_myr=((x['Open']*(1+profit_thr)-x['Open'])*num_units)-commission_price
        #when it went wrong, sell it off when it reaches x% loss or wait till it reaches eod
        elif x[profit_field]==0:
            max_loss_val=x['next_nlow']-x['Open']
            thr_loss_val=(x['Open']*(1-loss_thr))-x['Open']
            if max_loss_val < thr_loss_val:
                return_myr=thr_loss_val*num_units-commission_price
            else:
                return_myr= (x['next_nclose']-x['Open'])*num_units-commission_price
    return return_myr

def calculate_short_returns(x,profit_field,pred_field,alt_pred_field,commission_price,num_units,profit_thr=0.03,loss_thr=0.01):
    return_myr=0
    #when the prediction is zero, the model accumulates no returns
    if x[pred_field]==0:
        return_myr=0
    #when the model makes a prediction
    elif x[pred_field]==1:
        #when the opposite site prediction is true, do nothing
        if x[alt_pred_field]==1:
            return_myr=0
        elif x[profit_field]==1:
        #when it is correct, sell it off when it reaches max x% profit zone
            return_myr=((x['Open']-x['Open']*(1-profit_thr))*num_units)-commission_price
        #when it went wrong, sell it off when it reaches x% loss or wait till it reaches eod
        elif x[profit_field]==0:
            max_loss_val=x['Open']-x['next_nhigh']
            thr_loss_val=x['Open']-(x['Open']*(1+loss_thr))
            if max_loss_val < thr_loss_val:
                return_myr=thr_loss_val*num_units-commission_price
            else:
                return_myr=(x['Open']-x['next_nclose'])*num_units-commission_price
    return return_myr

def max_drawdown(X):
    mdd = 0
    peak = X[0]
    mdd_peak=0
    mdd_x=0
    for x in X:
        if x > peak:
            peak = x
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
            mdd_peak=peak
            mdd_x=x
    return round(mdd_peak,2),round(mdd_x,2),round(mdd,2)
