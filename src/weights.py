import numpy as np 
import tensorflow as tf 
import seaborn as sn 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib as matplotlib
from PIL import Image, ImageFont, ImageDraw

def normalize_columns_weights(weights_dict):
    from sklearn.preprocessing import MinMaxScaler

    scaled_dict = {}
    for i in weights_dict.keys():
        scaled_dict[i] = []
        for l in range(len(weights_dict[i])):
            tmp = weights_dict[i][l]
            if len(weights_dict[i][l].shape)==1:
                tmp = weights_dict[i][l].reshape(-1,1)
            sc = MinMaxScaler((-1,1))
            scaled_dict[i].append(sc.fit_transform(tmp))
    return scaled_dict

def weights_heatmap(weights_dict, epoch, layer, normalize =True):
    assert layer%2==0, 'Make sure you chose a weights matrix and not the biais vector'
    t = 1/(np.max(weights_dict[epoch][layer])-np.min(weights_dict[epoch][layer])) * (weights_dict[epoch][layer] - np.min(weights_dict[epoch][layer]))
    if normalize:
        sn.heatmap(t, cmap='coolwarm')
    else: 
        sn.heatmap(weights_dict[epoch][layer], cmap = 'coolwarm')
    plt.show()

def color_map_color(array, cmap_name='coolwarm', vmin=-1, vmax=1):
    
    array = np.asarray(array)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name) 
    if len(array.shape)==1:
        rgb = np.array(cmap(norm(array))[:,:3]) 
    else:
        rgb = np.array(cmap(norm(array))[:,:,:3])  # will return rgba, we take only first 3 so we get rgb
    return rgb

def create_colored_dict(weights_dict_enc):
    colored_dict_enc = {}
    epochs = len(weights_dict_enc.keys())
    for i in range(epochs):
        colored_dict_enc[i] = []
        for l in range(len(weights_dict_enc[i])):
            colored_dict_enc[i].append(color_map_color(weights_dict_enc[i][l]))
        #print(strbarwidth ,end = '')
    return colored_dict_enc

def interpolate_gif(weights_dict, block, layer, gifs_path):
    title_font = ImageFont.load_default()
    width = 256
    if np.min(weights_dict[0][layer].shape[:-1]) !=1:
        height = int((256/np.max(weights_dict[0][layer].shape[:-1]))*np.min(weights_dict[0][layer].shape[:-1]))
    else: 
        height=32

    
    images_list = [Image.fromarray((weights_dict[i][layer]*255).astype('uint8'), mode='RGB').resize((height, width)) for i in range(len(weights_dict))]
    draw_list = [ImageDraw.Draw(image).text((1,1), str(idx),font=title_font) for idx, image in enumerate(images_list)]
    images_list[0].save(gifs_path+
        f'{block}/layer_{layer}_shape_{weights_dict[0][layer].shape[:-1]}.gif',
        save_all=True,
        append_images=images_list[1:],
        duration = 16,
        loop=0)

