# Scalable-Model-Training-Pipeline-for-Time-Series-Analysis

This repository contains a scalable model training pipeline originally developed for the M5 competition hosted on Kaggle. It has been adapted for the purposes of the project to demonstrate scalability and efficient resource utilization during model training.

## Objectives

1. **Code Packaging**: Refactor the model training code into Python files with `main.py` as the entry point.

2. **Parallelization**: Enhance the pipeline to utilize multiple cores for parallel model training.

3. **Pipeline Definition**: Reorganize the code to define a machine learning pipeline, allowing sequential data processing steps culminating in model output.

## Data Description

[Link to Dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy)

The M5 dataset involves the unit sales of various products sold in the USA, organized in the form of grouped time series. More specifically, the dataset involves the unit sales of 139 products (instead of 3,049 in the original M5 competition), classified into 3 product categories (Hobbies, Foods, and Household) and 7 product departments, in which the above-mentioned categories are disaggregated.  The products are sold across ten stores, located in three States (CA, TX, and WI).


The historical data range from 2011-01-29 to 2016-05-22. Thus, the products have a (maximum) selling history of 1,913 days (test data of h=28 days not included). 

The M5 dataset consists of the following three (3) files:

File 1: “calendar.csv” <br>
Contains information about the dates the products are sold.

File 2: “sell_prices.csv” <br>
Contains information about the price of the products sold per store and date.  

File 3: “sales_train.csv”  <br>
Contains the historical daily unit sales data per product and store.

## Model Description
The following table lists the different levels at which model training could be performed. For example, at model level 10, it could be assumed that each item has different sales patterns but those patterns are similar across different stores, so models could be built per item. Since 139 items are being sold across 10 states, there will be 139 models (1 for each item) and each model will be trained on 10 time-series corresponding to the sale of the item in the 10 stores. At other model levels, there could be varying numbers of time series for training each model depending on the breakdown of the dataset as outlined in Figure 1 above.
Table 1: Number of models per level.
| Level | Model Level                                          | Number of models | Number of raw time series for training each model at this level |
|-------|------------------------------------------------------|------------------|----------------------------------------------------------------|
| 1     | Predict unit sales of all products, for all stores/states      | 1                | 1,390                                                            |
| 2     | Predict unit sales of all products, for each State              | 3                | variable                                                         |
| 3     | Predict unit sales of all products, for each store              | 10               | 139                                                              |
| 4     | Predict unit sales of all products, for each category           | 3                | variable                                                         |
| 5     | Predict unit sales of all products, for each department         | 7                | variable                                                         |
| 6     | Predict unit sales of all products, for each State and category | 9                | variable                                                         |
| 7     | Predict unit sales of all products, for each State and department| 21               | variable                                                         |
| 8     | Predict unit sales of all products, for each store and category | 30               | variable                                                         |
| 9     | Predict unit sales of all products, for each store and department| 70               | variable                                                         |
| 10    | Predict unit sales of product x, for all stores/states          | 139              | 10                                                               |
| 11    | Predict unit sales of product x, for each State                 | 417              | variable                                                         |
| 12    | Predict unit sales of product x, for each store                 | 1,390            | 1                                                                |


## Model Architecture
The forecasting model is built using a combination of convolutional neural network (CNN) layers and long short-term memory (LSTM) layers, designed to predict one step ahead in a time series. This architecture is particularly suited for handling the complexities of temporal data found in the M5 competition dataset.

### Model Details
1. Convolutional Layers: The model begins with convolutional layers (Conv1D), which apply filters to the time series data, capturing temporal dependencies. Each convolutional layer is followed by a max-pooling layer (MaxPooling1D) to reduce the dimensionality of the data and to abstract higher-level features.
2. LSTM Layers: After the convolutional layers, the model includes several LSTM layers which are designed to remember long-term dependencies in sequence data. The inclusion of batch normalization (BatchNormalization) helps in stabilizing the learning process and reducing the training time.
Dense Layer: The output of the LSTM layers is flattened and fed into a dense layer (Dense), which outputs the forecast for the next time step.

### Training
The model is compiled with the Mean Squared Error (mse) loss function and the Adam optimizer, which is an efficient variant of the stochastic gradient descent algorithm.
The model employs RootMeanSquaredError as a metric for evaluation.
Training involves an EarlyStopping callback to prevent overfitting, which monitors the validation loss and stops training if the model does not improve for a set number of epochs.
TensorBoard is utilized for logging and visualization purposes, allowing for monitoring of the model's performance and behavior during training.

# Getting started
## To create a virtual environment run the following commands in the terminal:

```
pip3 install virtualenv
```
```
virtualenv {name_of_environment}
```
```
source name_of_environment/bin/activate
```


## To install all requirements, run this command in the terminal:
```
pip install -r requirements.txt
```

The code has a src folder, with the classes for pipelines and the overall logic. There is a main file, which is the entry point to the code.

It can be run by using the command in the terminal:
```
python main.py
```

To create a subset of the data of size 1390.

main_sales_df= pd.read_csv('./data/sales_train_validation.csv')
main_sales=len(main_sales_df['item_id'].unique())
unique_item_ids = main_sales_df['item_id'].unique()
selected_item_ids = pd.Series(unique_item_ids).sample(n=139, random_state=42)
sales_df = main_sales_df[main_sales_df['item_id'].isin(selected_item_ids)]
sales_df.to_csv('./data/subset_train_validation.csv', index=False)


## To run the file, run this command in the terminal:
```
python main.py
```




