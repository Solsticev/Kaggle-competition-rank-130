import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

if __name__ == '__main__':
    

        df = pd.read_csv("train.csv")
        
        feature_train = np.array(df)
        target_train = feature_train[:,51]
        # print(id.shape)
        # print(target_train)
        feature_train = np.delete(feature_train, -1, axis = 1)
        feature_train = np.delete(feature_train, 0, axis = 1)
        # print(feature_train.shape)

        feature_test = pd.read_csv("test.csv")
        feature_test = np.array(feature_test)
        feature_test = np.delete(feature_test, 0, axis = 1)
        # print(feature_test.shape)
        
        # gbc = GradientBoostingClassifier(random_state=10) #scored 1.09278
        # gbc = GradientBoostingClassifier(random_state=10, n_estimators=200, max_depth=4) #scored 1.09021
        # gbc = GradientBoostingClassifier(random_state=10, n_estimators=400, max_depth=4) #scored  1.09200 training time 486.224s
        # gbc = GradientBoostingClassifier(random_state=10, n_estimators=200, max_depth=4, learning_rate = 0.3) #scored 1.10120 training time 208.823s
        # gbc = GradientBoostingClassifier(random_state=10, n_estimators=400, max_depth=4, learning_rate = 0.03) #scored 1.08960 training time 436.311s
        # gbc = GradientBoostingClassifier(random_state=10, n_estimators=400, max_depth=4, learning_rate = 0.05) #scored 1.09070 training time 890.452s
        # gbc = GradientBoostingClassifier(random_state=5, n_estimators=500, max_depth=4, learning_rate = 0.03) #scored 1.09113 training time 538.163s
        # gbc = GradientBoostingClassifier(random_state=10, n_estimators=800, max_depth=5, learning_rate = 0.03) #scored 1.09144 training time 1225.864s
        # gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10) #scored 1.08765 training time 86.096s
        # gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10) #scored 1.09661 training time 139.258s
        # gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.1, n_estimators=400, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10) #scored 1.08826 training time 256.975s
        # gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.05, n_estimators=400, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10) #scored 1.08779 training time 190.613s
        # gbc = GradientBoostingClassifier(max_depth=4, learning_rate=0.1, n_estimators=400, min_samples_leaf=40, max_features='sqrt', subsample=0.8, random_state=10) #scored 1.08808 training time 165.09s
        # gbc = GradientBoostingClassifier(learning_rate=0.01, n_estimators=400, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10) #scored 1.08901 training time 150.023s
        # gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, min_samples_leaf=20, max_features='sqrt', subsample=0.8, random_state=10, min_samples_split =1200) #scored 1.08769 training time 150.284s
        # gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, min_samples_leaf=20, max_features=9, subsample=0.8, random_state=10, min_samples_split =1200) #scored 1.08741 training time 200.941s
        # gbc = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400, min_samples_leaf=20, max_features=9, subsample=0.7, random_state=10, min_samples_split =1200) #scored 1.08710 training time 154.957s
        # gbc = GradientBoostingClassifier(learning_rate=0.02, n_estimators=2000, min_samples_leaf=20, max_features=9, subsample=0.7, random_state=10, min_samples_split =1200) #scored 1.08676 training time 806.268s
        gbc = GradientBoostingClassifier(learning_rate=0.01, n_estimators=4000, min_samples_leaf=20, max_features=9, subsample=0.7, random_state=10, min_samples_split =1200) #scored 1.08676 training time 982.156s
        

        

        gbc.fit(feature_train, target_train)

        id = []
        for i in range(100000, 150000):
            id.append(i)
        id = np.array(id)
        # print(id)

        pred = gbc.predict_proba(feature_test)
        pred = np.insert(pred, 0, values=id, axis=1)
        pd_data = pd.DataFrame(pred,columns=['id', 'Class_1','Class_2','Class_3', 'Class_4'])
        pd_data.to_csv(r'./_submit.csv')
        

