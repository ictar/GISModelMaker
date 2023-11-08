import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, cohen_kappa_score

import base64
from io import BytesIO

from . import const

class Validator:
    # TODO: export report to file?
    def report(self, Y_true, Y_pred, title, lulc_name, save_to):
        report = {}
        cm = confusion_matrix(Y_true, Y_pred, labels=lulc_name)
        print('Confusion Matrix: \n',cm)  
        # plot confusion matrix
        report['cm'] = {}
        np.set_printoptions(precision=2)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = lulc_name)
        fig, ax = plt.subplots(figsize=(10,10))
        display.plot(ax=ax, xticks_rotation='vertical')
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        report['cm']['display'] = f'<img src=\'data:image/png;base64,{encoded}\'>'
        #plt.show()

        # plot heatmap of confusion matrix
        cm_percent = cm/np.sum(cm) 
        fig = plt.figure(figsize=(7, 7), facecolor='w', edgecolor='k')  
        sns.set(font_scale=1.5)  
        sns.heatmap(cm_percent,  
                    xticklabels=lulc_name,  
                    yticklabels=lulc_name,  
                    cmap="YlGn",  
                    annot=True,  
                    fmt='.2%',  
                    cbar=False,  
                    linewidths=2,  
                    linecolor='black')  

        plt.title(title)  
        plt.xlabel('Predicted')  
        plt.ylabel('Actual')
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        report['cm']['percent_hm'] = f'<img src=\'data:image/png;base64,{encoded}\'>'
        #plt.show()
        # classfication report
        report['classfication_report'] = pd.DataFrame(classification_report(Y_true, Y_pred, output_dict=True)).transpose()
        print("\nClassification Report:\n", report['classfication_report'])

        # Others
        error_matrix1=pd.crosstab(pd.Series(Y_pred, name="Predicted"),pd.Series(Y_true, name="Reference"), dropna=False)
        cls_cat=error_matrix1.index.values
            
        #extract all columns values (classes of existing dataset)
        ref_cat=error_matrix1.columns.values
        #make union of index and column values
        cats=(list(set(ref_cat) | set(cls_cat)))
        #reindex error matrix so that it has missing columns and fill the emtpy cells with 0.00000001
        error_matrix=error_matrix1.reindex(index=cats, columns=cats, fill_value=0.00000001)
        error_matrix.index.name=error_matrix.index.name+"/"+error_matrix.columns.name
        error_matrix = error_matrix.rename(columns=const.LULC_Legends, index=const.LULC_Legends)
        # OUTPUT    
        diag_elem=np.diagonal(np.matrix(error_matrix))
        UA=(diag_elem/error_matrix.sum(axis=1))*(diag_elem>0.01) # Major_Diagonal/Row_Total
        PA=diag_elem/error_matrix.sum(axis=0)*(diag_elem>0.01) # Major_Diagonal/Column_Total
        OA=sum(diag_elem)/error_matrix.sum(axis=1).sum()
        KP = metrics.cohen_kappa_score(Y_true, Y_pred)
        PR = metrics.precision_score(Y_true, Y_pred, average=None)
        RE = metrics.recall_score(Y_true, Y_pred, average=None)
        F1 = metrics.f1_score(Y_true, Y_pred, average=None)#2*((PR*RE)/(PR+RE))
        error_matrix['UA']=UA.round(2)
        error_matrix['PA']=PA.round(2)
        error_matrix['Precision']=PR.round(2)#np.nan
        error_matrix['Recall']=RE.round(2)#np.nan
        error_matrix['F1']=F1.round(2)#np.nan
        report['error_matrix'] = error_matrix
        # Overall statistics
        overall_stat = pd.DataFrame(pd.Series({
            'Overall Accuracy': OA,
            'Precision': metrics.precision_score(Y_true, Y_pred, average='micro'),
            'Recall': metrics.recall_score(Y_true, Y_pred, average='micro'),
            'F1': metrics.f1_score(Y_true, Y_pred, average='micro'),
            "Cohen's Kappa": KP,
        }))
        report['overall_stat'] = overall_stat
        return report