def inference(para1,para2):
    
    fold_test = 0
    fold = 5   
    test_list = []
    for i in tqdm(range(fold)):
        df_train = df[df['kfold'] != i]
        df_test = df[df['kfold'] == i]
        model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs', C = para1, penalty = para2)
        model.fit(df_train[cols], df_train.quality)
        p = model.predict(df_test[cols]) 
        test_accuracy = cohen_kappa_score(df_test.quality, p, weights = "quadratic")
        fold_test += test_accuracy

    return fold_test / fold
    
 for i in tqdm((0.001, 0.01, 0.1, 1, 10, 100)):
    for n in tqdm(('l2', 'none')):
        accuracy = inference(i, n)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_parameters["C"] = i
            best_parameters["penalty"] = n
        else:
            pass
