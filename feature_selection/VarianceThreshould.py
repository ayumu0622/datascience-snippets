from sklearn.feature_selection import VarianceThreshold

def VarianceThreshould(df, df_test):
   
    #quality is target and kfold is fold column
    #df_test does have target columns but no kfold column.
    
    data = df.drop(["kfold", "quality"], axis = 1)
    var_thresh = VarianceThreshold(threshold = 0.01)
    var_thresh.fit(data)
    a = var_thresh.get_support()
    data = data.loc[:, a]
    df = pd.concat([data,df[["kfold", "quality"]]], axis = 1)
    df_te = df_test.drop(["quality"], axis = 1)
    df_te = df_te.loc[:, a]
    df_test = pd.concat([df_te,df_test[["quality"]]], axis = 1)
    
    return df, df_test
