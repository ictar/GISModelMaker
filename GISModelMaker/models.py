from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from . import const
    
def basic_model_generator(CLF):
    def basic_model(X, Y, clf_params):
        clf = CLF(**clf_params)
        clf = clf.fit(X, Y)
        return clf
    return basic_model

def grid_search_CV_generator(model):
    def grid_search_CV(X, Y, clf_params):
        m = model()
        clf = GridSearchCV(estimator=m, param_grid=clf_params, cv= 5)
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
}

def get(name):
    return _models.get(name)

