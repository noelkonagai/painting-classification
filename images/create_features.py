import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skimage.color as sc
from scipy import ndimage as nd
from skimage.morphology import watershed, disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.viewer import ImageViewer
from skimage import data as sd

pd.set_option('precision',10)

'''
opens image, converts to HSV, returns HSV array, height, width, total number of pixels
'''

def pre_process(path):
    img = sd.imread(path)
    img_hsv = sc.convert_colorspace(img, 'RGB', 'HSV')
    h = img_hsv.shape[0]
    w = img_hsv.shape[1]
    px = h * w
    
    return img_hsv, h, w, px

'''
support function for features 1-3
'''

def make_dataframe(img_hsv, h, w):
    height = len(img_hsv)
    width = len(img_hsv[0])
    hue = []
    saturation = []
    value = []
    
    for i in range(height):
        for j in range(width):
            hue.append(img_hsv[i][j][0])
            saturation.append(img_hsv[i][j][1])
            value.append(img_hsv[i][j][2])
            
    data = {'H': hue, 'S': saturation, 'V': value}
    hsv_df = pd.DataFrame(data=data)
    
    hue = np.array(hue)
    saturation = np.array(saturation)
    value = np.array(value)
    hue.shape = (h,w)
    saturation.shape = (h,w)
    value.shape = (h,w)
    
    return hsv_df, hue, saturation, value

'''
the first two features: average hue, average saturation
'''

def f1_2(hsv_df):
    return hsv_df.mean()['H'], hsv_df.mean()['S']


'''
support function for features 3 to 23, k is number of bins
'''
def make_bins(hsv_df, k):
    bins = [0]
    increment = 1/k
    for i in range(k):
        temp = bins[i] + increment
        bins.append(temp)
    hue_bins = pd.cut(hsv_df['H'], bins=bins, include_lowest=True).value_counts()
    return hue_bins


'''
count the num of hue bins that are most prominent, c is a multiplicatory component
'''
def f3(hue_bins, c):
    
    max_bins = max(hue_bins.tolist())
    count = 0
    
    for i in range(len(hue_bins)):
        if hue_bins[i] > c * max_bins:
            count +=1
            
    return count

'''
features 4 to 23, the dispersion of each hue component, returns as list
'''

def f4_23(hue_bins, px):
    hue_bins = hue_bins.sort_index().tolist()
    color_dispersion = []
    for i in range(len(hue_bins)):
        color_dispersion.append(hue_bins[i]/px)
          
    return color_dispersion

'''
performs water segmentation on images, which will be used for features
'''

def water_segmentation(path,hue,d1,d2,v,d3):

    image = img_as_ubyte(hue)
    image_original = sd.imread(path)
    image_gray = img_as_ubyte(sd.imread(path, as_grey=True))

    # denoise image
    denoised = rank.median(image, disk(d1))
    denoised_gray = rank.median(image_gray)

    # find continuous region (low gradient - where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(d2)) < v
    markers = nd.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(d3))

    # process the watershed
    labels = watershed(gradient, markers)
    
    return image, image_original, denoised, markers, gradient, labels

'''
displayes the segmentation process
'''

def display_water_segmentation(image, image_original, denoised, markers, gradient, labels):

    # display results
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
    ax = axes.ravel()

    ax[0].imshow(image_original, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title("Original")

    ax[1].imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
    ax[1].set_title("Local Gradient")

    ax[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
    ax[2].set_title("Markers")

    ax[3].imshow(image_original, cmap=plt.cm.gray, interpolation='nearest')
    ax[3].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest', alpha=.7)
    ax[3].set_title("Segmented")

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()
    
 #original configuration for water shedding
# display_water_segmentation(image, image_original, denoised, markers, gradient, labels)

'''
support function to count the largest segments
'''

def make_segment_df(labels):
    labels_df = pd.DataFrame(data=labels)
    unique, counts = np.unique(labels, return_counts=True)

    data = {'seg': unique, 'px': counts}
    segment_df = pd.DataFrame(data=data)
    segment_df = segment_df.sort_values(by = 'px', ascending = False)
    
    return segment_df

def filter_labels(labels, n):
    arr = np.copy(labels)
    boole = arr[arr > n] = 0
    boole = arr[arr < n] = 0
    boole = arr[arr == n] = 1
    
    return arr

def center_of_mass(arr, h, w):
    coordinates = nd.measurements.center_of_mass(arr)
    area = np.count_nonzero(arr)
    
    x_center = coordinates[0] / w
    y_center = coordinates[1] / h
    
    x = []
    y = []
    
    for i in range(h):
        for j in range(w):
            if arr[i][j] == 1:
                x.append(j)
                y.append(i)
                
    x = np.array(x)/w
    y = np.array(y)/h
    
    v1 = x.sum()/area ## in the original numbering this is f23-f25
    v2 = y.sum()/area ## f26-f28
    
    v3 = ( ((x - x_center) ** 2).sum() + ((y - y_center) ** 2).sum() ) / area ## f29-f31
    v4 = ( ((x - x_center) ** 3).sum() + ((y - y_center) ** 3).sum() ) / area ## f32-f34

    
    return v1, v2, v3, v4


'''
extract features 23 to 34
'''

def f24_35(df, lab, height, width):
    f24_f26 = []
    f27_f29 = []
    f30_f32 = []
    f33_f35 = []
    
    for i in range(3):
        seg_num = df['seg'][i]
        
        array = filter_labels(lab, seg_num)
        v1, v2, v3, v4 = center_of_mass(array, height, width)
        
        f24_f26.append(v1)
        f27_f29.append(v2)
        f30_f32.append(v3)
        f33_f35.append(v4)
        
    return f24_f26, f27_f29, f30_f32, f33_f35

'''
extracts features 36-50, returns them in a list format
'''

def f36_50(df, lab, hue, sat, val, h, w):
    f36_40 = []
    f41_45 = []
    f46_50 = []
    
    for i in range(5):
        seg_num = df['seg'][i]
        
        array = filter_labels(lab, seg_num)
        area = np.count_nonzero(array)
        
        hue_filtered = []
        sat_filtered = []
        val_filtered = []
        
        for i in range(h):
            for j in range(w):
                hue_filtered.append(hue[i][j] * array[i][j])
                sat_filtered.append(sat[i][j] * array[i][j])
                val_filtered.append(val[i][j] * array[i][j])
                
        f36_40.append(np.array(hue_filtered).sum() / area)
        f41_45.append(np.array(sat_filtered).sum() / area)
        f46_50.append(np.array(val_filtered).sum() / area)
        
    return f36_40, f41_45, f46_50