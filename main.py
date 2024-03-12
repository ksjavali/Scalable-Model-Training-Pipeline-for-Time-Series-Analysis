#importing libraries and dependancies from other files
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None
pd.set_option('display.float_format', '{:.2f}'.format)
from src.model_builder import train_baseline_model
from src.model_level import get_model_details_and_data_per_level
from src.data_preprocess import read_config_file
# from keras import backend as K
# import tensorflow as tf
import os
import concurrent.futures
from time import time
from src.pipeline_call import X_train_exo, y_train_exo, X_valid_exo, y_valid_exo,train_df,valid_df,n_products_stores_exo, n_products_stores


#main function
def main():
    #calling function to read the configuration file
    config = read_config_file('config.yaml')

    if config:
        epochs = config.get('epochs')  
        model_level = config.get('model_level') 

        print("Epoch:", epochs)
        print("Model Level:", model_level)
    else:
        print("Failed to read the config file.")
    n_training=28
    n_outputs = n_products_stores
    #function to get model details for the level and the data for the level
    model_details_and_data = get_model_details_and_data_per_level(model_level)
    # print("Model details len:",len(model_details_and_data))
    # print("Model details:",model_details_and_data)

    # Run training on multiple CPU cores
    num_cores = os.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")

    # Define the number of cores you want to use for parallel processing
    num_parallel_cores = min(num_cores, len(model_details_and_data))
    start=time()

    # Use ThreadPoolExecutor to train models in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_parallel_cores) as executor:
        futures = [executor.submit(train_baseline_model, X_train_exo, y_train_exo, X_valid_exo, y_valid_exo,epochs,model_details,train_df,valid_df,n_training, n_products_stores_exo,n_outputs) for model_details, _ , _ in model_details_and_data]
    end=time()

    print("time with process pool=",end-start)

    # Wait for all futures to complete
    for future in concurrent.futures.as_completed(futures):
        future.result()
    return True
    

#calling main function
if __name__=='__main__':
    main()

 