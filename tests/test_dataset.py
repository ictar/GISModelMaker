import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import unittest
import uuid
import os
import numpy as np
import rasterio as rio

from GISModelMaker.dataset import DataSet

class TestDataSet(unittest.TestCase):

    def _load_raster_layer(self, rpath):
        with rio.open(rpath) as ds:
            return ds.read()
        
    def test_gen_map_shape(self):
        # check if the shape is correct
        ds = DataSet(feature_names=['feature1', 'feature2'], label_name='label')
        target_path = r'./RGB2.byte.tif'
        ds.load_raster_data(target_path)
        trans = ds.flatten_target()
        trans_path = f'{uuid.uuid1()}.tf'
        ds.gen_map(trans, trans_path, bandcnt=trans.shape[-1])
        # check
        narr1 = self._load_raster_layer(target_path)
        narr2 = self._load_raster_layer(trans_path)
        self.assertTrue(np.array_equal(narr1, narr2, equal_nan=True))
        # clear
        os.remove(trans_path)


if __name__ == '__main__':
    unittest.main()