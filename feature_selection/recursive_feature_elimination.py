from sklearn.feature_selection import RFE

def RFEE(df, df_test):
    cols = list(df.columns)
    cols.remove('quality')
    cols.remove('kfold')
    X = df[cols]
    y = df.quality
    model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    rfe = RFE(estimator = model, verbose = 10,n_features_to_select = 210)
    rfe.fit(X,y)
    a = rfe.support_
    df = pd.concat([X.loc[:, a],df[["quality", "kfold"]]], axis = 1)
    df_test.drop("quality", axis = 1)
    t = df_test.drop("quality", axis = 1)
    df_test = pd.concat([t.loc[:, a], df_test[["quality"]]], axis = 1)
    
    return df, df_test
