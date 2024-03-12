from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


#class for splitting training and test sets
class SplitTrainTest(BaseEstimator,TransformerMixin):
    def __init__(self, n_forecast=28):
        self.n_forecast = n_forecast
        pass

    def fit(self, X=None):
        return self
    
    def transform(self, sales_df):
        train_df = sales_df.iloc[:,:-self.n_forecast].copy()
        valid_df = sales_df.iloc[:,-self.n_forecast:].copy()

        train_d_cols = [col for col in train_df.columns if col.startswith('d_')]
        fixed_cols = [col for col in train_df.columns if not col.startswith('d_')]
        valid_d_cols = [col for col in valid_df.columns if col.startswith('d_')]

        if not all([col in valid_df.columns for col in fixed_cols]):
            valid_df = pd.concat([train_df[fixed_cols],valid_df],axis=1,sort=False)

        return train_df, valid_df,fixed_cols,valid_d_cols