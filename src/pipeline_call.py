from sklearn.pipeline import Pipeline
from src.downcaster import DownCasting
from src.split_train_test import SplitTrainTest
from src.data_prep import InputOuputManipulation
from src.data_scaler import DualScaler
from src.shared import sales_df

#pipeline structure
pipeline = Pipeline(
   steps=[
       ("down_caster", DownCasting()),
       ("data_split", InputOuputManipulation(n_training=28, n_forecast=28)),
       ("min_max_scaler", DualScaler())
   ]
)

split_train_test=SplitTrainTest()
sales_df=pipeline.named_steps['down_caster'].transform(sales_df)
train_df, valid_df,fixed_cols,valid_d_cols=split_train_test.transform(sales_df)
# calendar_df=pipeline.named_steps['down_caster'].transform(calendar_df)
X_train_exo, y_train_exo, X_valid_exo, y_valid_exo, n_products_stores, n_products_stores_exo = pipeline.fit_transform(sales_df)