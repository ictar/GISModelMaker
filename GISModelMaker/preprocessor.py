from . import const
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, FastICA, NMF
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import f_classif

class Preprocesser:
    def __init__(self):
        # transformers
        self.transformers = {}
        self.transformer_factory = {
            # decomposition
            const.PREPRO_PCA: PCA(n_components=0.99),
            const.PREPRO_RBF_KERNEL_PCA: KernelPCA(n_components=const.MAX_COMPONENT, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1),
            # Factor Analysis
            const.PREPRO_FA: FactorAnalysis(n_components=const.MAX_COMPONENT),
            const.PREPRO_FICA: FastICA(n_components=const.MAX_COMPONENT),
            #const.PREPRO_NMF: NMF(n_components=const.MAX_COMPONENT),
            # feature selection
            #const.PREPRO_VT: VarianceThreshold(threshold=500),
            const.PREPRO_SKB_F1 : SelectKBest(f_classif, k=const.MAX_COMPONENT),
        }
        self.transf_names = None

    ## Feature selection
    ## normalize
    ## ref: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
    def feature_scaling(self, X, scaler=StandardScaler):
        print("Feature scaling using ", scaler)
        scaler = scaler()#.set_output(transform="pandas")
        return scaler.fit_transform(X)
    
    def register_transformer(self, transf_name, transf):
        print("Register Transformer ", transf_name)
        self.transformer_factory[transf_name] = transf
    
    def list_transformer_name(self):
        print(self.transformers.keys())
    def list_allowed_transformer(self):
        print(self.transformer_factory.keys())

    def describe_current_transformers(self):

        for transf_name in self.transf_names:
            pass

    def set_current_transformers(self, transf_names):
        self.transf_names = transf_names

    def run(self, X, Y=None, need_scale=True):
        X_trans = X
        if need_scale:
            X_trans = self.feature_scaling(X_trans)
        
        for transf_name in self.transf_names:
            if transf_name not in self.transformers:
                print(f"No {transf_name} exists, fit one")
                # note that some transformer doesn't used Y even if Y is passed
                self.transformers[transf_name] = self.transformer_factory[transf_name].fit(X_trans, Y)

            X_trans = self.transformers[transf_name].transform(X_trans)
            print(f"After transform by {transf_name}, the number of features was reduced from {X.shape} to {X_trans.shape}")

        return X_trans
    
    def describe(self):
        info = {}
        for name, p in self.transformers.items():
            info[name] = {}
            # common
            method = getattr(p, 'get_params', None)
            if method:
                info[name]['params'] = method()
            method = getattr(p, 'get_feature_names_out', None)
            if method:
                info[name]['feature_names_out'] = method()
            # optional 
            method = getattr(p, 'get_covariance', None)
            if method:
                info[name]['covariance'] = method()

        return info