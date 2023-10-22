# Import libraries
from ml_pipeline import processing, utils, training
from hyperopt import fmin, tpe, hp, anneal, Trials
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import gc

# Defining Parameters
data_path = 'input/credit_risk_data.csv'  # Path of data

# Non-feature Columns (not to be used as features in the model)
id_cols = ['User_id', 'emi_1_dpd', 'emi_2_dpd', 'emi_3_dpd', 'emi_4_dpd', 'emi_5_dpd', 'emi_6_dpd', 'max_dpd', 'yearmo']

# 1. Read data and drop columns
df = utils.process_data(data_path, ['gender'])
print("Data read and processed!")

# 2. Split data
train, val, hold_out = utils.data_split(df)
print("Data split done!")

# 3. Label creation
train = processing.create_label(train, dpd=60, months=3)
val = processing.create_label(val, dpd=60, months=3)
hold_out = processing.create_label(hold_out, dpd=60, months=3)
print("Label columns added to the dataframe")

# 4. Adding features and filling nulls
# - % Amount Paid as interest in past Loan Repayment
# - % of Loans defaulted in the last 2 years
train = processing.derived_features(train)
val = processing.derived_features(val)
hold_out = processing.derived_features(hold_out)
print("Feature engineering done!")

# 5. Categorical encoding
train, val, hold_out = processing.categorical_transform(train, val, hold_out, id_cols)
print("Categorical encoding done!")

# 6. Feature selection
train, val, hold_out = processing.select_features(train, val, hold_out, id_cols)
print("Features selected")

# 7. Model Hyperparameter Tuning
result_ho = pd.DataFrame()
i = 0
space = training.space

# Defining Objective Function, to be defined in the engine
# Returns the score to optimize
def objective(space):
    global i
    global result_ho

    # Creating Lightgbm DataFrame
    lgb_train = lgb.Dataset(train.drop(columns=id_cols), label=train.label)
    lgb_val = lgb.Dataset(val.drop(columns=id_cols), label=val.label)

    # Parameters of the model
    params = {
        'num_leaves': int(space['num_leaves']),
        'max_depth': int(space['max_depth']),
        'learning_rate': space['learning_rate'],
        'objective': 'binary',
        'metric': 'auc',
        "boosting": "gbdt",
        'feature_fraction': space['feature_fraction'],
        'max_bin': int(space['max_bin']),
        'min_data_in_leaf': int(space['min_data_in_leaf']),
        "min_data_in_bin": int(space['min_data_in_bin']),
        "bagging_freq": 20,
        "random_seed": 2019,
        "lambda_l1": space['lambda_l1'],
        "lambda_l2": space['lambda_l2'],
        'pos_bagging_fraction': space['pos_bagging_fraction'],
        'neg_bagging_fraction': space['neg_bagging_fraction'],
        'verbose': -1
    }

    evals_result = {}
    
    clf = lgb.train(params, lgb_train, 20000, valid_sets=lgb_val,
                valid_names='val',
                early_stopping_rounds=50,
                verbose_eval=False, evals_result=evals_result)
    gc.collect()

    result = pd.DataFrame(clf.params, index=[0])

    # Calculating AUC
    pred_train = clf.predict(train[clf.feature_name()])
    pred_val = clf.predict(val[clf.feature_name()])
    pred_hold_out = clf.predict(hold_out[clf.feature_name()])
    gc.collect()

    train_auc = roc_auc_score(train.label, pred_train)
    val_auc = roc_auc_score(val.label, pred_val)
    hold_out_auc = roc_auc_score(hold_out.label, pred_hold_out)
    gc.collect()

    score = (abs(train_auc - val_auc) + 1) / ((1 + val_auc) * (1 + val_auc))
    
    result["train_auc"] = train_auc
    result["val_auc"] = val_auc
    result["hold_out_auc"] = hold_out_auc
    result["train_test_diff"] = train_auc - val_auc
    result["n_estimators"] = clf.best_iteration
    result["score"] = score
    
    del clf
    
    result_ho = result_ho.append(result)
    result_ho.to_csv('output/hyperopt_results.csv', index=False)
    i = i + 1
    
    return score

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, rstate=np.random.default_rng(7))
print("Model Hyperparameter Tuning results saved as hyperopt_results.csv")

# 8. Take the best set of parameters
hyperopt_results = pd.read_csv('hyperopt_results.csv')
best_param_index = hyperopt_results.index[hyperopt_results['score'] == hyperopt_results['score'].min()].tolist()[0]
lgbm_params = dict(hyperopt_results.iloc[best_param_index, :19])
print("Got the best set of parameters!")

# 9. Training the model
clf = training.train_lgb(train, val, lgbm_params)
print("Model training done!")

# 10. Save Model
clf.save_model('output/model.txt', num_iteration=clf.best_iteration)
print("Model saved!")
