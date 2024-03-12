import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Create plots to visualize validation and prediction data from a model with an option to plot the training data as well
def plot_preds(train_df, valid_df, pred_df, calendar_df, plot_train_data=False):

  n_ts = len(train_df)
  if n_ts > 10:           # limiting the number of timeseries that can be plotted to reduce plotly runtime
    n_ts = 10
    train_df  = train_df.iloc[:10,:]
    print('Since there are more than 10 timeseries, plotting only top 10 as an example')
  
  fig = make_subplots(rows=n_ts, cols=1, subplot_titles=[id.replace('_validation','') for id in train_df['id']])

  for i in range(n_ts):

    train_ts_df = train_df[[col for col in train_df.columns if 'd_' in col]].iloc[i].reset_index().rename(columns={'index':'d'})
    valid_ts_df = valid_df[[col for col in valid_df.columns if 'd_' in col]].iloc[i].reset_index().rename(columns={'index':'d'})
    pred_ts_df  = pred_df [[col for col in pred_df.columns  if 'd_' in col]].iloc[i].reset_index().rename(columns={'index':'d'})

    train_ts_df = pd.merge(train_ts_df, calendar_df[['d','date']])
    valid_ts_df = pd.merge(valid_ts_df, calendar_df[['d','date']])
    pred_ts_df  = pd.merge(pred_ts_df, calendar_df[['d','date']])

    showlegend=True if i==0 else False

    if plot_train_data:
      fig.add_trace(go.Scatter(x=train_ts_df['date'], y=train_ts_df.iloc[:,1], name='training', line=dict(color='blue'),  showlegend=True if i==0 else False, legendgroup=str(i)), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=valid_ts_df['date'], y=valid_ts_df.iloc[:,1], name='validation', line=dict(color='black'), showlegend=True if i==0 else False, legendgroup=str(i)), row=i+1, col=1)
    fig.add_trace(go.Scatter(x=pred_ts_df['date'],  y=pred_ts_df.iloc[:,1], name='predictions', line=dict(color='green'), showlegend=True if i==0 else False, legendgroup=str(i)), row=i+1, col=1)
  
  fig.update_layout(height=n_ts*180, margin=dict(l=0,r=0,b=0,t=20), legend_tracegroupgap=150)
  fig.show()
