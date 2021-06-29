import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import log_loss

import gc

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations,callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers

from keras.models import Model


if __name__ == '__main__':

        df1 = pd.read_csv('train.csv')
        df2 = pd.read_csv('test.csv')
        sam = pd.read_csv('sample_submission.csv')
        X = df1.drop(columns = ['id','target'])
        y = df1['target']
        X_test = df2.drop(columns=['id'])

        # model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=4000, min_samples_leaf=20, max_features=9, subsample=0.7, random_state=10, min_samples_split =1200)#4960.549s 1.75379
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.8, n_estimators=60, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.80373
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.1, n_estimators=60, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.75312
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.01, n_estimators=600, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.75114
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.0025, n_estimators=2400, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.75079
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.0005, n_estimators=12000, min_samples_leaf=2100, max_features=8, subsample=0.8, random_state=10, min_samples_split =1690)# 1.75069
        # model = GradientBoostingClassifier(max_depth=6, learning_rate=0.0025, n_estimators=2400, min_samples_leaf=2100, max_features=8, subsample=0.7, random_state=10, min_samples_split =1690)# 1.75065
        
        # model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=800, max_features=9, subsample=0.7, random_state=10, min_samples_split =600) #1.7645
        # model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=1000, max_features='sqrt', subsample=0.8, random_state=10, min_samples_split =1000, min_samples_leaf=20) #1.76150
        # model = GradientBoostingClassifier()
        # model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=400,  max_features='sqrt', subsample=0.8, random_state=10, min_samples_split =600, min_samples_leaf=30) #498.787s 1.75533
        model = LogisticRegression() #1.82984, 32.857s
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

        model.fit(X, y)
    
        pred = model.predict_proba(X_test)
        sam[['Class_1','Class_2', 'Class_3', 'Class_4','Class_5','Class_6', 'Class_7', 'Class_8', 'Class_9']] = pred
        sam.to_csv(f'submit.csv',index=False)


# ============================ N N ================================ #

'''
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import log_loss

import gc

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations,callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import initializers

from keras.models import Model


N_FOLDS = 10
SEED = 2017
EPOCH = 60

train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")
submission = submission.set_index('id')

targets = pd.get_dummies(train['target'])

cce = tf.keras.losses.CategoricalCrossentropy()
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_custom_metric', min_delta=1e-05, patience=5, verbose=0,
    mode='min', baseline=None, restore_best_weights=True)
plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_custom_metric', factor=0.7, patience=2, verbose=0,
    mode='min')

def conv_model():
    
    conv_inputs = layers.Input(shape = (75))
    
    embed = layers.Embedding (input_dim = 354, 
                              output_dim = 7,
                              embeddings_regularizer='l2')(conv_inputs)
    
    embed = layers.Conv1D(12,1,activation = 'relu')(embed)        
    embed = layers.Flatten()(embed)
    hidden = layers.Dropout(0.3)(embed)
   
    hidden = tfa.layers.WeightNormalization(
                layers.Dense(units=32, activation ='selu', kernel_initializer = "lecun_normal"))(hidden)
    
    output = layers.Dropout(0.3)(layers.Concatenate()([embed, hidden]))
    output = tfa.layers.WeightNormalization(
    layers.Dense(units = 32, activation='relu',kernel_initializer = "lecun_normal"))(output) 
    
    output = layers.Dropout(0.4)(layers.Concatenate()([embed, hidden, output]))
    output = tfa.layers.WeightNormalization(
    layers.Dense(
                units = 32, activation = 'elu', kernel_initializer = "lecun_normal"))(output)
    
    conv_outputs = layers.Dense(
                units = 9, 
                activation ='softmax',
                kernel_initializer ="lecun_normal")(output)
    
    return Model(conv_inputs,conv_outputs)

oof_NN_a = np.zeros((train.shape[0],9))
pred_NN_a = np.zeros((test.shape[0],9))

def custom_metric(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-15, 1-1e-15)
    loss = K.mean(cce(y_true, y_pred))
    return loss

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (tr_idx, ts_idx) in enumerate(skf.split(train,train.iloc[:,-1])):
   
    X_train = train.iloc[:,1:-1].iloc[tr_idx]
    y_train = targets.iloc[tr_idx]
    X_test = train.iloc[:,1:-1].iloc[ts_idx]
    y_test = targets.iloc[ts_idx]

    K.clear_session()

    model_conv = conv_model()

    model_conv.compile(loss='categorical_crossentropy', 
                            optimizer = 'adam', 
                            metrics=custom_metric)
    model_conv.fit(X_train, y_train,
              batch_size = 256, epochs = EPOCH,
              validation_data=(X_test, y_test),
              callbacks=[es, plateau],
              verbose = 0)
   
    pred_a = model_conv.predict(X_test) 
    oof_NN_a[ts_idx] += pred_a 
    score_NN_a = log_loss(y_test, pred_a)

    pred_NN_a += model_conv.predict(test.iloc[:,1:]) / N_FOLDS 
 
score = log_loss(targets, oof_NN_a)

print("\n Final Score = ", score) 

pred_embedding = pred_NN_a

submission = pd.read_csv("sample_submission.csv")

submission[['Class_1','Class_2', 'Class_3', 'Class_4','Class_5','Class_6', 'Class_7', 'Class_8', 'Class_9']] = pred_embedding

submission.to_csv("submission.csv", index=False)
'''