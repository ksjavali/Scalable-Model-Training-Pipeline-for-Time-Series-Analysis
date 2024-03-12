import pandas as pd
import numpy as np
from src.data_preprocess import load


#location of the data
sales='data/subset_sales_train_validation.csv'
prices='data/sell_prices.csv'
calendar="data/calendar.csv"

#load the data using function
sales_df=load(sales)
prices_df=load(prices)
calendar_df=load(calendar)


