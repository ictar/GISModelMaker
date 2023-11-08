NaN = -9999
CLS_NODATA = 255

LULC_Legends = {101: '101 Bareland', 102: '102 Cropland', 103: '103 Forest',
                104: '104 Grassland', 105: '105 Built-up', 108: '108 Water'}
LULC_Legends_COLOR = {'#ffc425': (101, 'Bareland'), "#00ff83": (102, 'Cropland'), "#028900": (103, 'Forest'), "#adff00": (104, 'Grassland'), "#d11141": (105, 'Built-up'), "#00aedb": (108, 'Water')}

# preprocessor name
PREPRO_PCA = 'PCA'
PREPRO_RBF_KERNEL_PCA = 'KernelPCA (RBF)'
PREPRO_FA = 'FA'
PREPRO_FICA = 'FastICA'
PREPRO_NMF = 'NMF'

# Model name
MODEL_RANDOMFOREST = 'RandomForest'
MODEL_RANDOMFOREST_GS = 'RandomForest with GridSearch'
MODEL_SVC = 'SVM'
MODEL_SVC_GS = 'SVM with GridSearch'
# report template
report_tpl_path = 'report.tpl'