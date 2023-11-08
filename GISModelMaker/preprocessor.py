from . import const
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, FastICA, NMF

class Preprocesser:
    def __init__(self):
        # transformers
        self.transformers = {}
        self.transformer_factory = {
            const.PREPRO_PCA: PCA(n_components=0.99),
            const.PREPRO_RBF_KERNEL_PCA: KernelPCA(kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1),
            # Factor Analysis
            const.PREPRO_FA: FactorAnalysis(),
            const.PREPRO_FICA: FastICA(),
            const.PREPRO_NMF: NMF(),
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

    def set_current_transformers(self, transf_names):
        self.transf_names = transf_names

    def run(self, X, need_scale=True):
        X_trans = X
        if need_scale:
            X_trans = self.feature_scaling(X_trans)
        
        for transf_name in self.transf_names:
            if transf_name not in self.transformers:
                print(f"No {transf_name} exists, fit one")
                self.transformers[transf_name] = self.transformer_factory[transf_name].fit(X_trans)

            X_trans = self.transformers[transf_name].transform(X_trans)
            print(f"After transform by {transf_name}, the number of features was reduced from {X.shape} to {X_trans.shape}")

        return X_trans