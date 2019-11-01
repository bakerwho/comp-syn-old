import colorsys
import matplotlib.pyplot as plt
import numpy as np

def plot_hue_dist(distribution_dict):
    distribution_dict, h_dict, s_dict, v_dict, rgb_dict, entropy_dict, compressed_img_array_dict = get_image_structure(searchterms_list, home, img_dict)
    spacing = 10
    bins = np.arange(0,360+spacing,spacing)
    centers = (bins[:-1] + bins[1:]) / 2
    meanhsvcolor = colorsys.hsv_to_rgb(np.mean(h_dict[word],axis=0),
                                       np.mean(s_dict[word],axis=0),
                                       np.mean(v_dict[word],axis=0))
    meanrgbcolor = np.mean(np.array(rgb_dict[searchterms_list[0]]),axis=0)
    rgbcolors = []
    for i in range(len(centers)):
        rgbcolors.append(colorsys.hsv_to_rgb(centers[i]/360.,
                                             np.mean(s_dict[word],axis=0),
                                             np.mean(v_dict[word],axis=0)))
    f,ax = plt.subplots(1,1,figsize=(8,6))

    alpha = 0.17
    N = 360

    avg_rgb = np.mean(np.array(distribution_dict[word]),axis=0)

    ind = np.linspace(0,N,N/spacing) #the x locations for the groups
    width = spacing #the width of the bars

    p1 = ax.bar(ind, spacing*avg_rgb, width, color=rgbcolors)

    #ax.set_ylim(0,1.1)
    ax.set_title('{}'.format(word), fontsize=20, color=meanrgbcolor)
    ax.set_xticks(centers[::5])
    ax.set_xticklabels(centers[::5].astype(int), fontsize=16)
    #ax.set_yticks((0,0.2,0.4,0.6,0.8,1.0))
    #ax.set_yticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1'), fontsize=20)

    [t.set_color(i) for (i,t) in
     zip(rgbcolors[::5],ax.xaxis.get_ticklabels())]

    plt.ylabel(r'$\mathcal{P}(\rm{hue})$',fontsize=20)
    plt.subplots_adjust(wspace=0.25)
    plt.show()
