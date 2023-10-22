# Importing libraries
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Function to create default labels
def create_label(df, dpd, months):
    """Genrate label according to dpd in months,
    returns dataframe with label columns
    Parameters
    ----------
    df : DataFrame
    dpd : Int (30, 60, 90)
    months : Int (1,2,3,4,5,6)
    
    Returns
    -------
    df : DataFrame
    """
    try:
        months = ["emi_"+str(x)+"_dpd" for x in range(1, months+1)]
        df['label'] = np.where(df[months].max(axis = 1)>=dpd, 1, 0)
    except Exception as e:
        print(e)
    else:
        return df


# Features
def derived_features(df):
    """Create Some Features
    ----------
    df : DataFrame
    
    Returns
    -------
    df : DataFrame
    """
    try:
        df['interest_received_ratio'] = (df['interest_received']/df['total_payement']).replace([np.inf, -np.inf], 0).fillna(0)
        df['total_payement_per_loan'] = (df['total_payement']/df['number_of_loans']).replace([np.inf, -np.inf], 0).fillna(0)
        df['delinq_2yrs_ratio'] = (df['delinq_2yrs']/df['number_of_loans']).replace([np.inf, -np.inf], 0).fillna(0)
    except Exception as e:
        print(e)
    else:
        return df




# Categorical encoding - target encoding
class categorical_encoding:
    """Target Encoding of categorical variables
    input dataframe, categorical columns, label name, parameters of target_encoder
    """
    def __init__(self,params):
        """
        Parameters
        ----------
        params : Dict
        """
        self.params = params

    def fit(self, df, cat_cols, label):
        """Fitting Encoder
        Parameters
        ----------
        df : DataFrame
        cat_cols : List (Categorical columns)
        label : String
        """
        self.te = ce.target_encoder.TargetEncoder(**self.params)
        self.te.fit(df[cat_cols], df[label])
    
    def transform(self, d):
        """Transforming Data Encode and inplace transform categorical features
        Parameters
        ----------
        d : DataFrame
        
        Returns
        -------
        d : DataFrame
        """
        d = pd.concat([d.drop(columns = self.te.feature_names), self.te.transform(d[self.te.feature_names])], axis = 1)
        return d

def categorical_transform(train, val, hold_out, id_cols):
    '''
    categorical encoding on train, val, hold_out
    ---------
    train: DataFrame
    val: DataFrame
    hold_out: DataFrame
    id_cols: List

    Returns:
    Categorical encoded 
    train: DataFrame
    val: DataFrame
    hold_out: DataFrame
    '''
    try:
        cat_cols = train.drop(columns = id_cols).select_dtypes(include=['category', 'object']).columns
        params = {"verbose":0,
            "cols":None,
            "drop_invariant":False,
            "return_df":True,
            "handle_missing":'value',
            "handle_unknown":'value',
            "min_samples_leaf":5000,
            "smoothing":1}
        target_encoder = categorical_encoding(params)
        target_encoder.fit(train, cat_cols, 'label')

        train = target_encoder.transform(train)
        val = target_encoder.transform(val)
        hold_out = target_encoder.transform(hold_out)

    except Exception as e:
        print(e)
    else:
        return train, val, hold_out




# Feature Selection for Random Forest and Decision Tree
def random_forest_zero_importance(df, id_cols, label, params):
    """Finding Zero Importance features using random forest
    ----------
    df : DataFrame
    id_cols : List
    label : String
    params : Dict

    Returns
    -------
    zero_fi : List
    """
    rf = RandomForestClassifier(**params)
    rf.fit(df.drop(columns = id_cols).fillna(0), df['label'])
    fi = pd.DataFrame({"features":df.drop(columns = id_cols).columns, "importance":rf.feature_importances_})
    zero_fi = fi[fi.importance==0]['features']
    return zero_fi

def decision_tree_zero_importance(df, id_cols, label, params):
    """Finding Zero Importance features using decision tree
    ----------
    df : DataFrame
    id_cols : List
    label : String
    params : Dict

    Returns
    -------
    zero_fi : List
    """
    dt = DecisionTreeClassifier(**params)
    dt.fit(df.drop(columns = id_cols).fillna(0), df['label'])
    fi = pd.DataFrame({"features":df.drop(columns = id_cols).columns, "importance":dt.feature_importances_})
    zero_fi = fi[fi.importance==0]['features']
    return zero_fi

# combine both the functions for feature selection

def select_features(train, val, hold_out, id_cols):
    '''
    drops the common set of zero importance features from random forest and decision tree
    --------
    train: Dataframe
    val: DataFrame
    hold_out: DataFrame

    Returns:
    Features dropped from datasets
    train: Dataframe
    val: Dataframe
    hold_out: Dataframe
    '''
    try:
        rf_params = {"n_estimators":250, 'criterion':'entropy','verbose':False, 'n_jobs':25}
        rf_zero_imp = random_forest_zero_importance(train, id_cols, 'label', rf_params)
        dt_params = {}
        dt_zero_imp = decision_tree_zero_importance(train, id_cols, 'label', dt_params)
        drop_cols = list(set(rf_zero_imp) & set(dt_zero_imp))
        # remove the drop cols
        train = train.drop(drop_cols, axis=1)
        val = val.drop(drop_cols, axis=1)
        hold_out = hold_out.drop(drop_cols, axis=1)
    
    except Exception as e:
        print(e)

    else:
        return train, val, hold_out
        
