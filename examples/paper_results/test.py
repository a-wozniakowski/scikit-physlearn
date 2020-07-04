from physlearn.supervised.regression import Regressor

model = 'stackingregressor'
stack = dict(regressors=['mlpregressor', 'lgbmregressor'],
             final_regressor='mlpregressor')

reg = Regressor(regressor_choice=model, stacking_layer=stack)
reg.check_regressor
