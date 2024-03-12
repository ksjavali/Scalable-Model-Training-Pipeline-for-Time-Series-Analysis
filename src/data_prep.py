from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from src.shared import calendar_df

#class for creating exogenous data, train data, val data, etc
class InputOuputManipulation(BaseEstimator,TransformerMixin):
  def __init__(self, n_training, n_forecast):
    self.n_training=n_training
    self.n_forecast=n_forecast
  def fit(self, X):
    return self
  def transform(self, sales_df, y=None):
    # transposing data to merge exogenous features from calendar_df
    sales_t_df = sales_df[[col for col in sales_df.columns if col not in ['id','item_id','dept_id','cat_id','store_id','state_id']]].T.reset_index().rename(columns={'index':'d'})
    n_products_stores = sales_t_df[[col for col in sales_t_df.columns if 'd' not in str(col)]].shape[1]

    calendar_df['event_1_flag'] = calendar_df['event_name_1'].notna().astype(int)
    calendar_df['event_2_flag'] = calendar_df['event_name_2'].notna().astype(int)
    sales_t_exo_df = pd.merge(sales_t_df, calendar_df[['d','snap_CA','snap_TX','snap_WI','event_1_flag','event_2_flag']]).drop(columns=['d'])        # merging exogenous features to timeseries sales data
    sales_t_df = sales_t_df.drop(columns=['d'])
    n_products_stores_exo = sales_t_exo_df[[col for col in sales_t_exo_df.columns if 'd' not in str(col)]].shape[1]
    # transposing data to merge exogenous features from calendar_df
    sales_t_df = sales_df[[col for col in sales_df.columns if col not in ['id','item_id','dept_id','cat_id','store_id','state_id']]].T.reset_index().rename(columns={'index':'d'})
    n_products_stores = sales_t_df[[col for col in sales_t_df.columns if 'd' not in str(col)]].shape[1]
    calendar_df['event_1_flag'] = calendar_df['event_name_1'].notna().astype(int)
    calendar_df['event_2_flag'] = calendar_df['event_name_2'].notna().astype(int)
    sales_t_exo_df = pd.merge(sales_t_df, calendar_df[['d','snap_CA','snap_TX','snap_WI','event_1_flag','event_2_flag']]).drop(columns=['d'])        # merging exogenous features to timeseries sales data
    sales_t_df = sales_t_df.drop(columns=['d'])
    n_products_stores_exo = sales_t_exo_df[[col for col in sales_t_exo_df.columns if 'd' not in str(col)]].shape[1]

    # split sales data into train and validation
    train_sales_t_df = sales_t_df.iloc[:-self.n_forecast,:]
    valid_sales_t_df = sales_t_df.iloc[-self.n_forecast:,:]

    # split sales data merged with exogenous features into train and validation
    train_sales_t_exo_df = sales_t_exo_df.iloc[:-self.n_forecast,:]
    valid_sales_t_exo_df = sales_t_exo_df.iloc[-self.n_forecast:,:]

    return train_sales_t_df, valid_sales_t_df, train_sales_t_exo_df, valid_sales_t_exo_df, n_products_stores, n_products_stores_exo