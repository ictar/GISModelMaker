import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from . import const

# [TOTEST] Spectral Angle Mapper (SAM)
# reference: https://blog.csdn.net/qq_49425744/article/details/123445471
# cite: ‘Spectral Angle Mapper Algorithm for Remote Sensing Image Classification’ (2014) in. Available at: https://www.semanticscholar.org/paper/Spectral-Angle-Mapper-Algorithm-for-Remote-Sensing/c192da3f6560c0305926149e7f6324dab441201b (Accessed: 21 November 2023).

class SAM(object):
    def __int__(self):
        self.X = None
        self.y = None
        self.y_label = None 

    def fit(self, X, y):
        '''
        Fit the model to X.
        '''
        self.X = X
        self.y = y
        self.classes_ = np.unique(self.y)
        return self

    def predict(self, X):
        if X.ndim == 1:
            self.y_label = np.argmin(np.arccos(np.round(self.X.dot(X.reshape(-1,1))/(np.linalg.norm( \
                self.X, axis=1)*np.linalg.norm(X)).reshape(-1,1),5)))
        else:
            sita_f = self.X.dot(X.T)
            sita_m = np.linalg.norm(self.X, axis=1).reshape(-1,1).dot(np.linalg.norm(X.T, axis=0).reshape(1,-1))
            SITA = np.arccos(np.round(sita_f/sita_m, 5))
            self.y_label = np.argmin(SITA, axis=0)
        
        #print(f"SAM predict, y_label: {self.y_label}, \npredict result: {self.y[self.y_label]}")
        return np.asarray(self.y[self.y_label])
    
def basic_model_generator(CLF):
    def basic_model(X, Y, clf_params):
        clf = None
        if clf_params:
            clf = CLF(**clf_params)
        else:
            clf = CLF()
        clf = clf.fit(X, Y)
        return clf
    return basic_model

def grid_search_CV_generator(model):
    def grid_search_CV(X, Y, clf_params):
        m = model()
        clf = GridSearchCV(estimator=m, param_grid=clf_params, cv= 5, n_jobs=-1, verbose=2)
        clf.fit(X, Y)
        return clf

    return grid_search_CV


_models = {
    const.MODEL_RANDOMFOREST: basic_model_generator(RandomForestClassifier),
    const.MODEL_RANDOMFOREST_GS: grid_search_CV_generator(RandomForestClassifier),
    const.MODEL_SVC: basic_model_generator(SVC),
    const.MODEL_SVC_GS: grid_search_CV_generator(SVC),
    const.MODEL_NN_MLP: basic_model_generator(MLPClassifier),
    const.MODEL_NN_MLP_GS: grid_search_CV_generator(MLPClassifier),
    const.MODEL_SAM: basic_model_generator(SAM),
}

def get(name):
    return _models.get(name)