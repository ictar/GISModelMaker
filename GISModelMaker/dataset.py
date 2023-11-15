import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.plot import reshape_as_raster, reshape_as_image

from . import const

class DataSet:
    def __init__(self, feature_names, label_name='Thematic Class'):
        # target image
        self.target = None
        self.target_meta = None
        #self.target_shape = None
        self.target_nbands = None
        self.target_mask = None
        # samples
        self.sample_path = {}
        self.origin_train, self.origin_test = None, None
        self.X_train, self.Y_train = None, None
        self.X_test, self.Y_test = None, None
        # others
        self.label_name = label_name
        self.feature_names = feature_names
    
    # load data
    # ref: https://gis.stackexchange.com/questions/32995/fully-load-raster-into-a-numpy-array
    # ref: http://patrickgray.me/open-geo-tutorial/chapter_5_classification.html
    def load_raster_data(self, rpath, feature_names):
        arr = None
        with rio.open(rpath) as ds:
            arr = ds.read() # (bands, rows, columns)
            self.target_meta = ds.meta
            self.target_mask = np.where(arr[0,:,:]==ds.nodatavals[0], 0, 1)
            self.target_nbands = ds.count
            print(f"""[TARGET] path: {rpath}
            No values = {ds.nodatavals}
            Raw data shape = {arr.shape}""")
        # rasters are in the format [bands, rows, cols] whereas images are typically [rows, cols, bands]   
        self.target = reshape_as_image(arr) # (rows, columns, bands)
        # faltten the target: (rows*columns, bands)
        self.target = pd.DataFrame(
            self.target.reshape(-1, self.target_nbands),
            columns=feature_names)
        #self.target_shape = self.target.shape
        #self.target_nbands = self.target_shape[-1]
        print("After reshape, the data shape:", self.target.shape)
    

    def get_target(self):
        return self.target

    # load train/test
    ## max_per_class: maximum number of points per class, -1 means no restriction
    def split_train_test(self, sample_path, layer_name, max_per_class, test_perc=0.3):
        self.sample_path['raw'] = sample_path
        gdf = gpd.read_file(sample_path, layer=layer_name)
        # get feature and label
        gdf = gdf.fillna(const.NaN)
        if max_per_class != -1:
            gdf = gdf.groupby(self.label_name, group_keys=False).apply(lambda x: x.sample(min(max_per_class, len(x))))
            #print("After Resampling:\n", df.groupby(self.label_name)[self.label_name].count())
        tmp = sample_path.split(".")
        self.origin_test = gdf.sample(frac=test_perc)
        save_to = ".".join(tmp[:-1]) + "_test." + tmp[-1]
        self.origin_test.to_file(save_to, layer=layer_name, driver="GPKG")
        self.origin_train = gdf.drop(self.origin_test.index)
        save_to = ".".join(tmp[:-1]) + "_train." + tmp[-1]
        self.origin_train.to_file(save_to, layer=layer_name, driver="GPKG")
        print(f"""[Sample] path: {sample_path}, layer name: {layer_name}
        Shape = {gdf.shape}
        Columns = {gdf.columns}
        Label Stat:\n{gdf.groupby(self.label_name)[self.label_name].count()}
        Train sample Shape: {self.origin_train.shape}
        Test sample Shape: {self.origin_test.shape}""")
        
    def get_X_Y(self, sample_path, layer_name):
        gdf = gpd.read_file(sample_path, layer=layer_name)
        df = pd.DataFrame(gdf)
        # get feature and label
        df = df.fillna(const.NaN)
        X, Y = df[self.feature_names], df[self.label_name]
        print(f"==> Get X/Y from {sample_path} (layer name = {layer_name})\n==> Label Stat:\n{df.groupby(self.label_name)[self.label_name].count()}")
        return X, Y

    def get_training(self, sample_path, layer_name):
        if self.X_train is None:
            self.sample_path['train'] = (sample_path, layer_name)
            self.X_train, self.Y_train = self.get_X_Y( sample_path, layer_name)
        return self.X_train, self.Y_train
    def get_testing(self, sample_path, layer_name):
        if self.X_test is None:
            self.sample_path['test'] = (sample_path, layer_name)
            self.X_test, self.Y_test = self.get_X_Y( sample_path, layer_name)
        return self.X_test, self.Y_test

    def gen_map(self, data, save_to, bandcnt, dtype=None, nodata=None):
        if dtype is None: dtype = self.target_meta['dtype']
        if nodata is None: nodata = self.target_meta['nodata']
        # Reshape, original (rows*columns, bands)
        data = data.reshape((self.target_meta['height'], self.target_meta['width'], bandcnt))
        data = np.transpose(data, [2,0,1]) 
        data = np.where(self.target_mask==1, data, nodata)
        print(f"Generated map: shape = {data.shape}")
        with rio.open(save_to, 'w',
                      driver='GTIFF',
                      height=self.target_meta['height'], width=self.target_meta['width'],
                      count=bandcnt,
                      dtype=dtype,
                      crs=self.target_meta['crs'], transform=self.target_meta['transform'],
                      nodata=nodata
                     ) as out:
            out.write(data[:bandcnt, :, :])
    
    def describe_dataset(self, to_html=True):
        stats = {}
        if self.X_train is not None:
            stats['train'] = self.X_train.describe(include='all')
        if self.X_test is not None:
            stats['test'] =  self.X_test.describe(include='all')
        if self.target is not None:
            stats['target'] = pd.DataFrame(self.get_target()).describe(include='all')
        if to_html:
            for k in stats:
                stats[k] = stats[k].to_html()

        return stats