from sklearn.base import BaseEstimator, TransformerMixin
from src.data_preprocess import get_input_output_slices_exo
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#class for scaling data
class DualScaler(BaseEstimator, TransformerMixin):
    def __init__(self, feature_range=(0, 1), n_training=28, n_forecast=28):
        self.scaler_main = MinMaxScaler(feature_range=feature_range)
        self.scaler_exo = MinMaxScaler(feature_range=feature_range)
        self.n_training = n_training
        self.n_forecast = n_forecast

    def fit(self, inputs):
        return self

    def transform(self, inputs):
        train_sales_t_df, valid_sales_t_df, train_sales_t_exo_df, valid_sales_t_exo_df, n_products_stores, n_products_stores_exo = inputs
        # scaling train/validation data without exogenous features
        train_sales_t_df.columns = train_sales_t_df.columns.astype(str)
        valid_sales_t_df.columns  = valid_sales_t_df.columns.astype(str)
        train_sales_t_df_scaled = self.scaler_main.fit_transform(train_sales_t_df)
        valid_sales_t_df_scaled  = self.scaler_main.transform(valid_sales_t_df)

        # scaling train/validation data with exogenous features
        train_sales_t_exo_df.columns = train_sales_t_exo_df.columns.astype(str)
        valid_sales_t_exo_df.columns  = valid_sales_t_exo_df.columns.astype(str)
        train_sales_t_exo_df_scaled = self.scaler_exo.fit_transform(train_sales_t_exo_df)
        valid_sales_t_exo_df_scaled  = self.scaler_exo.transform(valid_sales_t_exo_df)

        # Prepare training data slices
        print('\nTraining data preparation:')
        X_train_exo, y_train_exo = get_input_output_slices_exo(train_sales_t_exo_df_scaled, 0, train_sales_t_exo_df_scaled.shape[0]-self.n_training, self.n_training, n_products_stores)

        # Prepare validataion data slices
        print('\nValidation data preparation:')
        concat_train_valid_exo_scaled = np.concatenate((train_sales_t_exo_df_scaled, valid_sales_t_exo_df_scaled), axis=0)
        X_valid_exo,  y_valid_exo  = get_input_output_slices_exo(concat_train_valid_exo_scaled, concat_train_valid_exo_scaled.shape[0]-self.n_training-self.n_forecast, concat_train_valid_exo_scaled.shape[0]-self.n_forecast, self.n_training, n_products_stores)

        # Convert training and validation data slices to numpy arrays since providing them as a list to tensorflow throws an error
        X_train_exo = np.array(X_train_exo)
        y_train_exo = np.array(y_train_exo)
        X_valid_exo  = np.array(X_valid_exo)
        y_valid_exo  = np.array(y_valid_exo)

        return X_train_exo, y_train_exo, X_valid_exo, y_valid_exo, n_products_stores, n_products_stores_exo

    def inverse_transform(self, scaled_data):
        # Use the scaler_main to inverse transform the main data
        return self.scaler_main.inverse_transform(scaled_data)
