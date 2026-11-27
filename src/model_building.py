from sklearn.ensemble import RandomForestRegressor


def model_build(X_train,X_test,y_train,y_test):


     #ModelBuilding

     model = RandomForestRegressor().fit(X_train,
                                          y_train)   

     return model