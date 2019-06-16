''' understand the usage of hyperopt package

Tree Parzen Estimators: adjusts probability density based on target function evaluation history, uses bayesian approach to update
posterior hyperparameters distribution

Three main components:

- Space: collection of parameters and their pre defined spaces
- Function: takes in a parametrized space and returns a value to minimize
- Fmin

'''
from hyperopt import tpe,fmin, hp, Trials
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll.stochastic import sample as ho_sample
from hyperopt.pyll import scope as ho_scope

#trials = MongoTrials('mongo://localhost:27017/foo_db/jobs',
#                     exp_key='exp1')

# provides a stateful placeholder for parameter optimization history
trials = Trials()

def func(args):
    x,y=args
    f = x**2 + y**2
    return f

space = {'x':hp.uniform('x',-1,1),'y':hp.uniform('y',-2,3)}

best_params=fmin(func,space,algo=tpe.suggest,max_evals=50,trials=trials)
print(best_params)

## example for sampling parameters from a defined hp space
hp_space = {
    'n_estimators': ho_scope.int(hp.quniform('n_estimators', low=100, high=300, q=25)),
    'criterion': hp.choice('criterion', ['gini', 'entropy']),
    'max_features': hp.uniform('max_features', low=0.25, high=0.75)}

ho_sample(hp_space)
