import ngboost
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ARDRegressionRandomForestRegressor

features = ['Volume_L1', 'OBV_L1', 'Volume_L2', 'EMA_W_L1', 'Open_stationary_L3', 'MA_L1', 'EMA_L1', 'MA_2_L3', 'MA_W_L2', 'MA_W_L3', 'Volume_L3', 'ATR_L2', 'OBV_L3', 'EMA_W_L3', 'ADOSC_L3', 'High_minus_Low_L3', 'MA_2_L2', 'STD_2_L1', 'TRANGE_L3', 'ADOSC_L1', 'High_stationary_L3', 'STD_W_L2', 'STD_L3', 'Low_stationary_L3', 'STD_L2', 'Low_stationary_L1', 'STD_2_L3', 'rr_L1', 'rr_L2', 'rr_L3']


def mp_cv(train_valid_setup):
    train = train_valid_setup[0]
    test = train_valid_setup[1]
    
    if train_valid_setup[2] == 'EN':
        model = ElasticNet()
    elif train_valid_setup[2] == "RF":
        model = RandomForestRegressor(n_estimators=25, max_depth=3)
    elif train_valid_setup[2] == "DT":
        model = DecisionTreeRegressor(max_depth=3)
    
    ngb = ngboost.NGBRegressor(
            Dist=ngboost.distns.Laplace,
            Score=ngboost.scores.LogScore,
            Base=model,
            n_estimators=500,
            learning_rate=0.01,
            minibatch_frac=1.0,
            col_sample=1.0,
            verbose=False,
            verbose_eval=500,
            tol=0.0001,
            random_state=2021)
    ngb.fit(train[features], train["rr"])
    Y_dists = ngb.pred_dist(test[features])
    
    return [Y_dists.dist.ppf(0.01)[0], Y_dists.dist.ppf(0.025)[0]]
    