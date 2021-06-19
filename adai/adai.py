import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


if __name__ == '__main__':

        df = pd.read_csv("train.csv")
        feature_train = np.array(df)
        target_train = feature_train[:,76]
        feature_train = np.delete(feature_train, -1, axis = 1)
        feature_train = np.delete(feature_train, 0, axis = 1)
        feature_test = pd.read_csv("test.csv")
        feature_test = np.array(feature_test)
        feature_test = np.delete(feature_test, 0, axis = 1)

        # model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=4000, min_samples_leaf=20, max_features=9, subsample=0.7, random_state=10, min_samples_split =1200)#4960.549s 1.75379
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.8, n_estimators=60, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.80373
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.1, n_estimators=60, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.75312
        model = GradientBoostingClassifier(max_depth=6, learning_rate=0.01, n_estimators=600, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.75114
        # model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=800, max_features=9, subsample=0.7, random_state=10, min_samples_split =600) #1.7645
        # model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000, max_features='sqrt', subsample=0.8, random_state=10, min_samples_split =1000, min_samples_leaf=20) #1.76150
        # model = GradientBoostingClassifier()
        # model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400,  max_features='sqrt', subsample=0.8, random_state=10, min_samples_split =600, min_samples_leaf=30) #498.787s 1.75533
        # model = LogisticRegression() #1.82984, 32.857s
        # model = GaussianNB() #13.13148, 6.745s
        
        '''
        param_test1 = {'n_estimators':range(20,81,10)}
        gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=10,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10), 
                                param_grid=param_test1, scoring='neg_log_loss',cv=5, n_jobs=-1)
       
        gsearch1.fit(feature_train, target_train)
        print(gsearch1.best_params_)
        print(gsearch1.best_score_)
        # output:{'n_estimators': 50} -1.755762303105438
        '''
        
        '''
        param_test2 = {'max_depth':range(3,11,1)}
        gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(n_estimators=50 ,learning_rate=0.1, min_samples_split=300,
                                  min_samples_leaf=10,max_features='sqrt', subsample=0.8,random_state=10), 
                                param_grid=param_test2, scoring='neg_log_loss',cv=5, n_jobs=-1)
       
        gsearch2.fit(feature_train, target_train)
        print(gsearch2.best_params_)
        print(gsearch2.best_score_)
        # output: {'max_depth': 6} -1.7529371056648926
        '''
        
        '''
        param_test3 = {'min_samples_split':range(10,2000,20)}
        gsearch3 = RandomizedSearchCV(estimator = GradientBoostingClassifier(max_depth=6, n_estimators=50 ,learning_rate=0.1,
                                  min_samples_leaf=10,max_features='sqrt', subsample=0.8,random_state=10), 
                                param_distributions=param_test3, scoring='neg_log_loss',cv=5, n_jobs=-1)
       
        gsearch3.fit(feature_train, target_train)
        print(gsearch3.best_params_)
        print(gsearch3.best_score_)
        # output: {'min_samples_split': 1690} -1.7521175591774736
        '''
        
        '''
        param_test4 = {'min_samples_leaf':range(10,2000,20)}
        gsearch4 = RandomizedSearchCV(estimator = GradientBoostingClassifier(min_samples_split=1690, max_depth=6, n_estimators=50 ,learning_rate=0.1,
                                  min_samples_leaf=10,max_features='sqrt', subsample=0.8,random_state=10), 
                                param_distributions=param_test4, scoring='neg_log_loss',cv=5, n_jobs=-1, n_iter=15)
       
        gsearch4.fit(feature_train, target_train)
        print(gsearch4.best_params_)
        print(gsearch4.best_score_)
        # output: {'min_samples_leaf': 1950} -1.7500687198400509
        '''
        
        '''
        param_test4 = {'min_samples_leaf':range(1500,8000,100)}
        gsearch4 = RandomizedSearchCV(estimator = GradientBoostingClassifier(min_samples_split=1690, max_depth=6, n_estimators=50 ,learning_rate=0.1,
                                  min_samples_leaf=10,max_features='sqrt', subsample=0.8,random_state=10), 
                                param_distributions=param_test4, scoring='neg_log_loss',cv=5, n_jobs=-1, n_iter=15)
       
        gsearch4.fit(feature_train, target_train)
        print(gsearch4.best_params_)
        print(gsearch4.best_score_)
        # output: {'min_samples_leaf': 2100} -1.7499949968071256
        '''
        
        '''
        param_test5 = {'max_features':range(0,75,4)}
        gsearch5 = RandomizedSearchCV(estimator = GradientBoostingClassifier(min_samples_leaf=2100, min_samples_split=1690, max_depth=6, n_estimators=50 ,learning_rate=0.1,
                  subsample=0.8,random_state=10), 
                  param_distributions=param_test5, scoring='neg_log_loss',cv=5, n_jobs=-1, n_iter=15)
       
        gsearch5.fit(feature_train, target_train)
        print(gsearch5.best_params_)
        print(gsearch5.best_score_)
        # output: {'max_features': 8} -1.7499949968071256
        '''
        
        '''
        param_test6 = {'max_features':range(0,30,1)}
        gsearch6 = RandomizedSearchCV(estimator = GradientBoostingClassifier(min_samples_leaf=2100, min_samples_split=1690, max_depth=6, n_estimators=50 ,learning_rate=0.1,
                  subsample=0.8,random_state=10), 
                  param_distributions=param_test6, scoring='neg_log_loss',cv=5, n_jobs=-1, n_iter=15)
       
        gsearch6.fit(feature_train, target_train)
        print(gsearch6.best_params_)
        print(gsearch6.best_score_)
        # output: {'max_features': 8} -1.7499949968071256
        '''

        


        model.fit(feature_train, target_train)
        id = []
        for i in range(200000, 300000):
            id.append(i)
        id = np.array(id)
        # print(id)

        pred = model.predict_proba(feature_test)
        pred = np.insert(pred, 0, values=id, axis=1)
        pd_data = pd.DataFrame(pred,columns=['id', 'Class_1','Class_2','Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 'Class_9',])
        pd_data.to_csv(r'./submit.csv')
