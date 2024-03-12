from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, BatchNormalization, Dense
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
import numpy as np
import pandas as pd
from src.visualize import plot_preds
from keras.callbacks import EarlyStopping, TensorBoard
from src.shared import calendar_df
from src.pipeline_call import pipeline,fixed_cols,valid_d_cols





def train_baseline_model(X_train_exo, y_train_exo, X_valid_exo, y_valid_exo,epochs, model_details,train_df,valid_df,n_training, n_products_stores_exo,n_outputs):
    # Baseline model summary
    print(model_details)
    #baseline_model = build_baseline_model()
    baseline_model = build_baseline_model(n_training, n_products_stores_exo,n_outputs)
    baseline_model.summary()
    

        # Train baseline model with early stopping and tensorboard callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    baseline_model_history = baseline_model.fit(X_train_exo, y_train_exo, epochs=epochs, batch_size=100, validation_data=(X_valid_exo,y_valid_exo), callbacks = [es, TensorBoard(log_dir="../results/baseline/tb_logs")])
    baseline_model_pred_df = evaluate_model(baseline_model, X_valid_exo,n_training,n_products_stores_exo,fixed_cols,valid_d_cols,train_df,pipeline)
    #plot predictions
    plot_preds(train_df, valid_df, baseline_model_pred_df, calendar_df)
    # Save baseline model
    baseline_model.save(f'./models/baseline_model_{model_details}.keras')



# Create baseline NN model using CNN-LSTM architecture that makes one step ahead predictions
def build_baseline_model(n_training, n_products_stores_exo,n_outputs):
    
    baseline_model = Sequential()

    conv1_filters = 64
    baseline_model.add(Conv1D(name='conv1', kernel_size=7, strides=1, padding="causal", activation="relu", filters=conv1_filters, input_shape=(n_training, n_products_stores_exo)))
    baseline_model.add(MaxPooling1D(name='pool1'))

    baseline_model.add(Conv1D(name='conv2', kernel_size=7, strides=1, padding="causal", activation='relu', filters=int(baseline_model.get_layer('conv1').output.shape[2]/2)))
    baseline_model.add(MaxPooling1D(name='pool2'))

    lstm1_units = 256
    baseline_model.add(LSTM(name='lstm1', units=lstm1_units, return_sequences=True))
    baseline_model.add(BatchNormalization(name='norm1'))

    baseline_model.add(LSTM(name='lstm2', units=int(baseline_model.get_layer('lstm1').output.shape[2]/2), return_sequences=True))
    baseline_model.add(BatchNormalization(name='norm2'))

    baseline_model.add(LSTM(name='lstm3', units=int(baseline_model.get_layer('lstm2').output.shape[2]/2)))
    baseline_model.add(BatchNormalization(name='norm3'))

    baseline_model.add(Dense(name='dense', units=n_outputs))

    learning_rate = 0.001
    opt_adam = Adam(clipvalue=0.5, learning_rate=learning_rate)

    baseline_model.compile(loss='mse', optimizer=opt_adam, metrics=[RootMeanSquaredError()])
    
    return baseline_model

# Compute WRMSSE for NN models
def evaluate_model(model, X_valid,n_training,n_products_stores_exo,fixed_cols,valid_d_cols,train_df,pipeline):

    y_valid_pred=[]
    for X in X_valid:
        y_valid_pred_day = model.predict(X.reshape(1,n_training,n_products_stores_exo))
        y_valid_pred_day = pipeline.named_steps['min_max_scaler'].inverse_transform(y_valid_pred_day)
        y_valid_pred.append(y_valid_pred_day)

    y_valid_pred_df = pd.DataFrame(np.array([y_valid_pred[i].reshape(-1,) for i in range(len(y_valid_pred))]).T, columns=valid_d_cols)
    if not all([col in y_valid_pred_df.columns for col in fixed_cols]):
      y_valid_pred_df = pd.concat([train_df[fixed_cols],y_valid_pred_df],axis=1,sort=False)

    return y_valid_pred_df
