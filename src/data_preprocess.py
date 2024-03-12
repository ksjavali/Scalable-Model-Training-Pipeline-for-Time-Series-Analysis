import pandas as pd
import yaml


def load(file_path):
    return pd.read_csv(file_path)

#function to read the configuration file
def read_config_file(file_path):
    with open(file_path, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            return None

#funtion to get input and output slices
def get_input_output_slices_exo(data_array, input_slice_start_min, input_slice_start_max, n_training, n_products_stores):
    
    X = []
    y = []
    
    print('Sample input and output slices for 1 step prediction')
    for i in range(input_slice_start_min, input_slice_start_max):
        if i<=input_slice_start_min+2 or i>input_slice_start_max-4:            # print a few slices at the start and end for debugging
            print('Input:',range(i,i+n_training),',  Output:',i+n_training)
        X.append(data_array[i:i+n_training])
        y.append(data_array[i+n_training,:n_products_stores])
    
    return X, y

