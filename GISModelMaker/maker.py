import os
import pickle
import time
from datetime import datetime
import base64
from io import BytesIO

import jinja2

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
import rioxarray as rxr
import rasterio as rio

from . import const
from .dataset import DataSet
from .preprocessor import Preprocesser
from . import models
from .validator import Validator

class Maker:

    def __init__(self,
        feature_names, label_name, sample_path, max_per_class=-1 # dataset
        ):
        # dataset
        self.ds = DataSet(feature_names, label_name)
        ## load samples
        if isinstance(sample_path, tuple):
            self.ds.split_train_test(sample_path[0], sample_path[1], max_per_class)
        if isinstance(sample_path, dict):
            if "train" in sample_path:
                self.ds.get_training(sample_path['train']['path'], sample_path['train']['layer'])
            if "test" in sample_path:
                self.ds.get_testing(sample_path['test']['path'], sample_path['test']['layer'])

        # preprocessor
        self.preprocessor = Preprocesser()
        # models, e.g., classifiers
        self.models = {}
        # validator
        self.valid = Validator()

    # map generalization using clf
    def gen_pred_map(self, target_path, model_name, save_to):
        self.ds.load_raster_data(target_path)
        # preprocess target data
        X_trans = self.preprocessor.run(self.ds.flatten_target())
        # save original to see if they are same
        # save transformed features
        feature_path = ".".join(save_to.split(".")[:-1]) + "_features." + save_to.split(".")[-1]
        self.ds.gen_map(X_trans, feature_path, bandcnt=X_trans.shape[-1])
        # predict
        cls_pred = self.models[model_name].predict(X_trans)
        # generate
        self.ds.gen_map(cls_pred, save_to, bandcnt=1, dtype='uint8', nodata=const.CLS_NODATA)

        return self.show_map(feature_path)

    def show_map(self, mpath, need_show=False):
        with rio.open(mpath) as src:
            count = min(src.count, const.MAX_COMPONENT)+1
            fig, ax = plt.subplots(count//2, 2, figsize=(10, count*10//4))
            for i in range(1, count):
                rio.plot.show((src, i), ax=ax[(i-1)//2, (i-1)%2], cmap='viridis', title=f'Band {i}', transform=src.transform)
            if not need_show:
                print("Generate Feature Images instead of showing")
                tmpfile = BytesIO()
                fig.savefig(tmpfile, format='png')
                encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
                return f'<img src=\'data:image/png;base64,{encoded}\'>'

            plt.show()
            

    def show_pred_map(self, mpath, title, legend_labels):
        if not mpath: return None
        r_chm = rxr.open_rasterio(mpath, masked=True).squeeze()
        # Define the colors you want
        cmap = ListedColormap(list(legend_labels.keys()))
        # Define a normalization from values -> colors
        bins = [0]
        bins.extend(sorted([v[0] for v in legend_labels.values()]))

        norm = colors.BoundaryNorm(bins, len(legend_labels))
        fig, ax = plt.subplots()
        chm_plot = ax.imshow(r_chm,
                            cmap=cmap,
                            norm=norm)

        ax.set_title(title)

        patches = [Patch(color=color, label=label[-1])
                for color, label in legend_labels.items()]

        ax.legend(handles=patches,
                bbox_to_anchor=(1.35, 1),
                facecolor="white")

        ax.set_axis_off()
        # save to base64
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        #plt.show()
        return f'<img src=\'data:image/png;base64,{encoded}\'>'

    # main workflow  
    def run(self, preprocess, model_name, model_param, save_report_path, target_path=None, save_target_path=None):
        time_stat = {}
        # preprocess
        start_time = time.time()
        self.preprocessor.set_current_transformers(preprocess)
        print(f"[{datetime.now()}] Begin to Preprocess {preprocess}")
        X_train = self.preprocessor.run(self.ds.X_train.values, self.ds.Y_train)
        print(f"[{datetime.now()}] End to Preprocess")
        time_stat['preprocessing'] = time.time() - start_time
        start_time = time.time()

        # process
        print(f"[{datetime.now()}] Begin to Process {model_name} using parameter {model_param}")
        self.models[model_name] = models.get(model_name)(X_train, self.ds.Y_train, model_param)
        print(f"[{datetime.now()}] End to Process, model information: {self.describe_model(model_name)}")
        time_stat['train'] = time.time() - start_time
        start_time = time.time()

        # validate
        print(f"[{datetime.now()}] Begin to Validate")
        X_test = self.preprocessor.run(self.ds.X_test)
        Y_test_pred = self.models[model_name].predict(X_test)
        report = self.valid.report(
            self.ds.Y_test, Y_test_pred,
            model_name, self.models[model_name].classes_,
            save_report_path)
        print(f"[{datetime.now()}] End to Validate")
        time_stat['validate'] = time.time() - start_time
        start_time = time.time()

        used_features_img = None
        # generate predict map
        if target_path and save_target_path:
            print(f"[{datetime.now()}] Begin to generate predicted map")
            used_features_img = self.gen_pred_map(target_path, model_name, save_target_path)
            print(f"[{datetime.now()}] Emd to generate predicted map")
        time_stat['map_gen'] = time.time() - start_time
        start_time = time.time()

        # generate report
        
        with open(save_report_path, 'w') as f:
            env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(__file__)))
            report_temp = env.get_template(const.report_tpl_path)
            report_temp_out = report_temp.render(
                dataset = self.ds,
                target_path=target_path,
                prepro_method=preprocess,
                prepro_param=None,
                used_pp_info=self.preprocessor.describe(),
                used_feature_number=X_train.shape[-1],
                used_features=used_features_img,
                model_name=model_name,
                model_param=model_param,
                model_info=self.describe_model(model_name),
                ds_stats=self.ds.describe_dataset(),
                error_matrix=report['error_matrix'].to_html(),
                overall_stat=report['overall_stat'].to_html(),
                clsf_report=report['classfication_report'].to_html(),
                cm_display=report['cm']['display'],
                cm_percent_heatmap=report['cm']['percent_hm'],
                prepro_time=time_stat['preprocessing'],
                train_time=time_stat['train'],
                evaludate_time=time_stat['validate'],
                map_time=time_stat['map_gen'],
                map=self.show_pred_map(save_target_path, "Preprocess: "+"+".join(preprocess)+", Model: " + model_name, const.LULC_Legends_COLOR)
            )
            f.write(report_temp_out)

    def set_preprocessors(self, pps):
        for k, pp in pps.items():
            self.preprocessor.register_transformer(k, pp)

    def describe_model(self,name):
        m = self.models[name]
        info = {'name': name}
        method = getattr(m, 'get_params', None)
        if method:
            info['params'] = method()
        # GridSearchCV
        #info['cv_results'] = getattr(m,'cv_results_', None)
        info['best_params'] = getattr(m, 'best_params_', None)
        # Basic

        return info
    def dump_object(self, fpath):
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)