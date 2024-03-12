from src.pipeline_call import sales_df

#groupbys is defined here, adding the "for_all" column for level 1
sales_df['for_all']='all'
groupbys = ('for_all', 'state_id', 'store_id', 'cat_id', 'dept_id',['state_id', 'cat_id'],  
            ['state_id', 'dept_id'], ['store_id', 'cat_id'],['store_id', 'dept_id'], 'item_id', 
            ['item_id', 'state_id'], ['item_id', 'store_id'])

def get_model_details_and_data_per_level(level):
    if isinstance(groupbys[level-1], list):
        list_groupby = groupbys[level-1]  
    else:
        list_groupby = [groupbys[level-1]]
    
    # Group the data and get the count for each group
    grouped_data = sales_df.groupby(list_groupby).count().reset_index()
    
    # Rename the count column
    grouped_data = grouped_data.rename(columns={'id':'number of raw time series for training each model at level '+str(level)})
    
    # Store both model details and corresponding subsets of data
    model_details_and_data = []
    for idx, group in grouped_data.iterrows():
        # Extract the group information
        group_info = group[list_groupby]
        
        # Filter the original dataframe to get the subset of data corresponding to this group
        subset_data = sales_df
        for col, value in zip(list_groupby, group_info):
            subset_data = subset_data[subset_data[col] == value]
        
        if 'for_all' in subset_data.columns:
            subset_data.drop(columns=['for_all'], inplace=True)
        
        # Store the model details along with the subset of data
        model_details_and_data.append((group_info, group['number of raw time series for training each model at level '+str(level)], subset_data))
    
    return model_details_and_data

