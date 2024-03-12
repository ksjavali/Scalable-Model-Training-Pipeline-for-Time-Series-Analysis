from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


#class for down casting
class DownCasting(BaseEstimator,TransformerMixin):
  def __init__(self,X=None, y=None):
    pass
  def fit(self,X=None, y=None):
    return self
  def transform(self, df):
    start_mem = df.memory_usage().sum() / 1024**2
    cols_float = df.select_dtypes('float').columns
    cols_int = df.select_dtypes('integer').columns
    df[cols_float] = df[cols_float].apply(pd.to_numeric, downcast='float')
    df[cols_int] = df[cols_int].apply(pd.to_numeric, downcast='integer')
    if 'date' in df.columns:
      df['date'] = pd.to_datetime(df['date'])
    end_mem = df.memory_usage().sum() / 1024**2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df





    

    




