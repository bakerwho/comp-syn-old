# data helper code

# DataCollector

import os
import PIL
from PIL import Image
import numpy as np
from collections import defaultdict

jzazbz_map = np.load('./jzazbz_array.npy', encoding = 'latin1')

def rgb_to_jzazbz(rgb_array):
    # rgb_array.shape = (col_pixel_count, row_pixel_count, #channels)
    assert isinstance(rgb_array, np.ndarray)
    d1, d2 = rgb_array.shape[0], rgb_array.shape[1]
    jzazbz_array = np.zeros((d1, d2, 3))
    for i in range(rgb_array.shape[0]):
        for j in range(rgb_array.shape[1]):
            r, g, b = (rgb_array[i][j][k] for k in range(3))
            # NOTE: RGB each must be >=0 and <=255
            jzazbz_array[i,j] = jzazbz_map[r][g][b][:3]
    return jzazbz_array

class ImageData():
    def __init__(self, **kwargs):
        self.rgb_dict = defaultdict(lambda : None)
        self.jzazbz_dict = defaultdict(lambda : None)
        self.labels_list = []
        self.dims = None

    def load_image_dict_from_subfolders(self, path, label=None, compress_dims=(300,300)):
        assert os.path.isdir(path)
        compress_dims = self.dims if self.dims else compress_dims
        self.dims = compress_dims
        path = os.path.realpath(path)
        folders = os.listdir(path)
        for folder in folders:
            fp = os.path.join(path, folder)
            assert os.path.isdir(fp)
            self.load_image_dict_from_folder(fp, label=label, compress_dims=compress_dims)
            self.store_jzazbz_from_rgb(label)
        self.labels_list = list(self.rgb_dict.keys())


    def load_image_dict_from_folder(self, path, label=None, compress_dims=(300,300), compute_jzazbz=True):
        assert os.path.isdir(path)
        compress_dims = self.dims if self.dims else compress_dims
        self.dims = compress_dims
        path = os.path.realpath(path)
        label = label or path.split('/')[-1]
        files = os.listdir(path)
        imglist = []
        arraylist = []
        for file in files:
            fp = os.path.join(path, file)
            img = None
            try:
                img = self.load_rgb_image(fp, compress_dims=compress_dims)
            except ValueError:
                continue
            if img is not None:
                imglist.append(img)
        if compute_jzazbz:
            self.store_jzazbz_from_rgb(label)
        self.rgb_dict[label] = imglist
        self.labels_list = list(self.rgb_dict.keys())

    def load_rgb_image(self, path, compress_dims=None):
        fmts = ['.jpg', '.jpeg', '.png', '.bmp']
        if os.path.isfile(path) and any([fmt in path.lower() for fmt in fmts]):
            try:
                img_raw = PIL.Image.open(path)
                if compress_dims:
                    assert len(compress_dims)==2
                    img_raw = img_raw.resize((compress_dims[0],compress_dims[1]),
                                                PIL.Image.ANTIALIAS)
                img_array = np.array(img_raw)[:,:,:3]
                assert len(img_array.shape)==3 and img_array.shape[-1]==3
                return img_array
            except:
                return None
                pass

    def store_jzazbz_from_rgb(self, labels=None):
        if labels:
            labels = labels if isinstance(labels, list) else [labels]
        else:
            labels = list(self.rgb_dict.keys())
        for label in labels:
            if label and label in self.rgb_dict.keys():
                try:
                    self.jzazbz_dict[label] = [rgb_to_jzazbz(rgb) for rgb in self.rgb_dict[label]]
                except:
                    pass

    def print_labels(self):
        self.labels_list = list(self.rgb_dict.keys())
        print(self.labels_list)

"""
class ImageDownload(self):
    build_image(input_string)
    build_request(image, annotation = ['label', 'web'])
    get_annotations(request)
    chunks(l, n)
    divide_chunks(l, n)
    run_google_vision(myImageList)
    write_to_json(to_save, filename)
    download_images(keywords, n)
    filter_imgs_w_google_scoring(img_json, DLpath, top_n)
    get_filtered_img_set(filtered_dict, home)
    get_imgs(searchterms_list, home)
    get_responses(keywords, n, path, filename, use_filter=False)
"""
