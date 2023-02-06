X = df[cols]
y = df.quality

def my_custom_loss_func(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights = "quadratic")

classifier = RandomForestClassifier(n_jobs = -1)
param_grid = {"n_estimators" : [100, 200, 300, 400, 500], "max_depth":[1,2,5,7,11,15], 
             "criterion" : ["gini", "entropy"]}


model = model_selection.GridSearchCV(estimator = classifier, param_grid = param_grid, scoring = make_scorer(my_custom_loss_func, greater_is_better=True), verbose=10, n_jobs=1,
                                    cv = 5)
model.fit(X,y)
print(f"Best score: {model.best_score_}")
      
print("Best parameters set:")
best_parameters = model.best_estimator_.get_params()
for param_name in sorted(param_grid.keys()):
      print(f"\t{param_name}: {best_parameters[param_name]}")
