# analysis code

import numpy as np
import scipy.stats
import time
import matplotlib.colors as mplcolors
import compsyn as cs

class ImageAnalysis():
    def __init__(self, image_data):
        #assert isinstance(image_data, compsyn.ImageData)
        self.image_data = image_data

    def compute_color_distributions(self, labels, color_rep=['jzazbz', 'hsv', 'rgb'], spacing=36):
        dims = self.image_data.dims
        labels = labels if isinstance(labels, list) else [labels]
        self.jzazbz_dist_dict, self.hsv_dist_dict = {}, {}
        self.rgb_ratio_dict, self.rgb_dist_dict = {}, {}
        color_rep = [i.lower() for i in color_rep]
        if 'jzazbz' in color_rep:
            self.jzazbz_dist_dict = {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    print("\nlabel {} does not exist".format(key))
                    continue
                if key not in self.image_data.jzazbz_dict.keys():
                    self.image_data.store_jzazbz_from_rgb(key)
                jzazbz, dist_array = [], []
                imageset = self.image_data.jzazbz_dict[key]
                for i in range(len(imageset)):
                    jzazbz.append(imageset[i])
                    dist = np.ravel(np.histogramdd(np.reshape(imageset[i][:,:,:],(dims[0]*dims[1],3)),
                                          bins=(np.linspace(0,0.167,3),np.linspace(-0.1,0.11,3),
                                               np.linspace(-0.156,0.115,3)), density=True)[0])
                    dist_array.append(dist)
                self.jzazbz_dist_dict[key] = dist_array
        if 'hsv' in color_rep:
            self.h_dict, self.s_dict, self.v_dict = {}, {}, {}
            self.hsv_dist_dict = {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    print("\nlabel {} does not exist".format(key))
                    continue
                imageset = self.image_data.rgb_dict[key]
                dist_array, h, s, v = [], [], [], []
                for i in range(len(imageset)):
                    hsv_array = mplcolors.rgb_to_hsv(imageset[i]/255.)
                    dist = np.histogram(360.*np.ravel(hsv_array[:,:,0]),
                                        bins=np.arange(0,360+spacing,spacing),
                                        density=True)[0]
                    dist_array.append(dist)
                    h.append(np.mean(np.ravel(hsv_array[:,:,0])))
                    s.append(np.mean(np.ravel(hsv_array[:,:,1])))
                    v.append(np.mean(np.ravel(hsv_array[:,:,2])))
                self.hsv_dist_dict[key] = dist_array
                self.h_dict[key], self.s_dict[key], self.v_dict[key] = h, s, v
        if 'rgb' in color_rep:
            self.rgb_ratio_dict, self.rgb_dist_dict = {}, {}
            for key in labels:
                if key not in self.image_data.labels_list:
                    print("\nlabel {} does not exist".format(key))
                    continue
                imageset = self.image_data.rgb_dict[key]
                rgb = []
                dist_array = []
                for i in range(len(imageset)):
                    r = np.sum(np.ravel(imageset[i][:,:,0]))
                    g = np.sum(np.ravel(imageset[i][:,:,1]))
                    b = np.sum(np.ravel(imageset[i][:,:,2]))
                    tot = 1.*r+g+b
                    rgb.append([r/tot,g/tot,b/tot])
                    dist = np.ravel(np.histogramdd(np.reshape(imageset[i],(dims[0]*dims[1],3)),
                                          bins=(np.linspace(0,255,3),np.linspace(0,255,3),
                                               np.linspace(0,255,3)), density=True)[0])
                    dist_array.append(dist)
                self.rgb_ratio_dict[key] = rgb
                self.rgb_dist_dict[key] = dist_array

    def cross_entropy_between_images(self, symmetrized=True):
        #needswork
        rgb_dict = self.image_data.rgb_dict
        entropy_dict = {}
        entropy_dict_js = {}
        for key in rgb_dict:
            entropy_array = []
            entropy_array_js = []
            for i in range(len(rgb_dict[key])):
                for j in range(len(rgb_dict[key])):
                    if symmetrized == True:
                        mean = (rgb_dict[key][i] + rgb_dict[key][j])/2.
                        entropy_array.append((scipy.stats.entropy(rgb_dict[key][i],rgb_dict[key][j])+scipy.stats.entropy(rgb_dict[key][j],rgb_dict[key][i]))/2.)
                        entropy_array_js.append((scipy.stats.entropy(rgb_dict[key][i],mean) + scipy.stats.entropy(rgb_dict[key][j],mean))/2.)
                    else:
                        entropy_array.append(scipy.stats.entropy(rgb_dict[key][i],rgb_dict[key][j]))
            entropy_dict[key] = entropy_array
            entropy_dict_js[key] = entropy_array_js
        return entropy_dict, entropy_dict_js

    def cross_entropy_between_labels(self, symmetrized=True):
        #needswork
        rgb_dict = self.image_data.rgb_dict
        mean_rgb_dict = {}
        for key in rgb_dict:
            mean_rgb_array = np.mean(np.array(rgb_dict[key]),axis=0)
            mean_rgb_dict[key] = mean_rgb_array
        labels_entropy_dict = {}
        labels_entropy_dict_js = {}
        color_sym_matrix = []
        color_sym_matrix_js = []
        for word1 in words:
            row = []
            row_js = []
            for word2 in words:
                if symmetrized == True:
                    mean = (mean_rgb_dict[word1] + mean_rgb_dict[word2])/2.
                    entropy = (scipy.stats.entropy(mean_rgb_dict[word1],mean_rgb_dict[word2])+scipy.stats.entropy(mean_rgb_dict[word2],mean_rgb_dict[word1]))/2.
                    entropy_js = (scipy.stats.entropy(mean_rgb_dict[word1],mean) + scipy.stats.entropy(mean_rgb_dict[word2],mean))/2.
                else:
                    entropy = scipy.stats.entropy(mean_rgb_dict[word1],mean_rgb_dict[word2])
                row.append(entropy)
                row_js.append(entropy_js)
                labels_entropy_dict[word1 + word2] = entropy
                labels_entropy_dict_js[word1 + word2] = entropy_js
            color_sym_matrix.append(row)
            color_sym_matrix_js.append(row_js)
        return labels_entropy_dict, color_sym_matrix, labels_entropy_dict_js, color_sym_matrix_js

    def get_composite_image(self, labels=None, compress_dim=300):
        compressed_img_dict = {}
        img_data = self.image_data.rgb_dict
        if not labels:
            labels = img_data.keys()
        for label in labels:
            compressed_img_array = np.zeros((compress_dim,compress_dim,3))
            for n in range(len(img_data[label])):
                if np.shape(img_data[label][n]) == (compress_dim, compress_dim, 3):
                    for i in range(compress_dim):
                        for j in range(compress_dim):
                            compressed_img_array[i][j] += img_data[label][n][i][j]/(1.*len(img_data[label]))
            compressed_img_dict[label] = compressed_img_array
        return compressed_img_dict

    #get_filtered_img_set(filtered_dict, home) # doesn't return


"""
WordAnalysis
    get_random_wordlist(N)
    get_synsets(wordlist)
    expandTree(wordlist)
    get_tree_structure(tree, home)
    get_tree_structure_simp(wordlist, home)
    get_sym_mat_tree(tree_data)
    get_sym_mat_simp(synsets_simp)
    write_to_json(to_save, filename)
    get_tree_data(wordlist, home, get_trees)
    get_word_heatmap(tree_data, home)
    get_wordnet_data(wordlist,home)
    expandTree(wordlist)
    get_branching_factor(wordlist)
    get_precision_basic(wordlist, use_syn = False, syndict={})
    get_precision_and_branching(super_dict, sub_dict, avgsuperprec, avgsubprec, avgsuperbranch, avgsubbranch)
"""
