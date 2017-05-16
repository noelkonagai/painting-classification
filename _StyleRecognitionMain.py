import urllib, pickle, re, json, os, requests, matplotlib, h5py, cv2
import scipy.io as sio
import numpy as np
from scipy import misc
from PIL import Image
from skimage.io import imread
from sklearn.externals import joblib

def download(name, U_r_l):
    
    with open(str(name), 'wb') as handle:
        print("Downloading from__:"+str(U_r_l))
        response = requests.get(U_r_l, stream = True)
                
        if not response.ok:
            print(response)
        for block in response.iter_content(1024):
            if not block:
                break
            handle.write(block)
    print('Out of DOWNLOAD')    


def convert_json_to_lists(labels_list):
    counter = 0
    artists = list()
    artistIDs = list()
    
    for n in range(75,len(labels_list)-1):
        
        line = labels_list[n]
        try:
            linestr = line.decode()
            linelist = linestr.split(',')
            if len(linelist) >= 10:
                url = linelist[4]
                artistID = linelist[0]
                artist_name = linelist[1][1:-1]
                if len(artist_name)!=0:
                
                    artists.append(artist_name)
                    artistIDs.append(artistID)
        except UnicodeDecodeError:
            print("UDE in start")
    path = 'C:\\Users\\Artem\\Google Drive\\All\\Study Materials (DRPBX)\\Machine Learning\\_StyleAnalyserProject\\ver 0\\dataset\\wikiart\\wikiart\\meta\\'
    genre_url = dict()                                   
    genre_id =dict()
    for nameind in range(len(artists)):
        name = artists[nameind]
        pathname = path+name+'.json'
        urls_for_genre = list()
        id_for_genre = list()
        try:
            with open(pathname, 'r') as f:
                data = json.load(f)
            for painting in data:
                
                urls_for_genre.append(painting['image'])
                id_for_genre.append(painting['contentId'])
                genre = painting['genre']
                if genre!= "NULL":
                    genre_url[genre] = urls_for_genre
                    genre_id[genre] = id_for_genre
                    counter +=1
        except FileNotFoundError:
            print('ERROR File Not Found____'+pathname+'\n\n')
        except UnicodeDecodeError:
            print("ERROR UNICODE DECODE ERROR____"+pathname+'\n\n')
            #dict[painting['genre']] = urls_for_genre
    pickle.dump( genre_url, open( "genre_url.p", "wb" ))   
    pickle.dump( genre_id, open( "genre_id.p", "wb" )   ) 
    print(counter)        
    print("checkpoint")

def convert_img_2array(name):
    
    try:
        #img = Image.open(name+'.jpg')
        im = cv2.imread(name+'.jpg')
        #print("SHAPE is: ___", arr.shape)
        if len(arr.shape)==2:
            w, h = arr.shape
            newarr = np.zeros((w,h,3))
            
            print("IS GRAYSCALE")
            for i in range(len(im)):
                for j in range(len(im[i])):
                    temp = im[i][j]
                    for k in range(len(newarr[i][j])):
                        newarr[i][j][k] = temp
            print(str(newarr.shape))            

        #arr =np.array(img)
        #arr = list(img.getdata())
            return newarr
        else:
            return im
    except FileNotFoundError:
        print('ERROR File Not Found____'+str(name)+'\n\n')
        return None;

def display_RGBarray(name):
    print("RGB array looks like this:")
    img = convert_img_2array(name)
    
    print(img)
    print("Displaying Image as Array")
    
def display_HSVarray(name):
    print("HSV array looks like this:")
    img = convert_img_2array(name)
    hsv = convert_to_hsv(img)
    print(hsv)
    
def convert_to_hsv(array):
    return matplotlib.colors.rgb_to_hsv(array)
    
def nolabels_list(genre_id):
    arr = list()
    for key, value in genre_id.items():
        arr.extend(value)
    return arr    

def convert_all_to_HSV(imagesID):
    path = '/Users/noel/Desktop/wikiart/images'
    hsvpath = '/Users/noel/Desktop/wikiart/images/hsv_noel'
    hsv_images = list()
    cnt = 0
    print(len(imagesID))
    for ID in imagesID:
        
        
        rgb = convert_img_2array(path+str(ID))
        if(rgb ==  None):
            
            continue
        else:
            cnt +=1
            print("____IMAGE NUMBER___"+str(cnt))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #print(hsv)
            pickle.dump(hsv, open(hsvpath+str(ID)+'.p', 'wb'))
            #hsv_images.append(hsv)
    return hsv

def average_hue(imageID):
    hsvpath = '/Users/noel/Desktop/wikiart/images/hsv'
    cnt = 0
    total = 0
    image = open(hsvpath+str(imageID)+'.p', 'rb')

    for line in image:
        for pixel in line:
            cnt+=1
            total += pixel[0]
    return total/cnt

def average_saturation(imageID):
    hsvpath = 'C:\\Users\\Artem\\Google Drive\\All\\Study Materials (DRPBX)\\Machine Learning\\_StyleAnalyserProject\\ver 0\\dataset\\wikiart\\wikiart\\images\\hsv\\'
    cnt = 0
    total = 0
    image = open(hsvpath+str(imageID)+'.p', 'rb')
    for line in image:
        for pixel in line:
            cnt+=1
            total += pixel[1]
    return total/cnt
 
#preprocess for feature 3- number of quantized hues


def largest_hue(imageID):
    hsvpath = 'C:\\Users\\Artem\\Google Drive\\All\\Study Materials (DRPBX)\\Machine Learning\\_StyleAnalyserProject\\ver 0\\dataset\\wikiart\\wikiart\\images\\hsv\\'
    
    image = open(hsvpath+str(imageID)+'.p', 'rb')
    large = 0
    for line in image:
        for pixel in line:
            if large < pixel[0]:
                large = pixel[0]
            
    return large

#FEATURE NUMBER 3 - NUMBER OF QUANTIZED HUES 

#returns a number in range of 20 since we use 20 bins

# NEEDS OPTIMIZATION
def largest_hist(bins):
    return max(bins)


def bins(hsv):
    q = [0]*20
    large = 0
    
    #bins list is only used to set borders for 20 different bins
    # it looks like this: [18, 36, 54, 72, 90, 108, 126, 144, 162, 180, 198, 216, 234, 252, 270, 288, 306, 324, 342, 360]
    
    bins = [0]*20
    for i in range(360//20+2):
        bins[i] = (1+i)*360//20
    print(bins)
    for line in hsv:
        for pixel in line:
            hue = pixel[0]
            
            for b in bins:
                if hue <= b and hue> bins[bins.index(b)-1]:
                    q[bins.index(b)]+=1
    return q

def feature_3(hsv):
    number = 0;
    bins = bins(hsv)
    for j in bins:
        if j > 0.1 * largest_hist(bins):
            number+=1
    return number

def imgsize(hsv):
    x = hsv.shape
    return x[0]*x[1]

#20 features formed into a list - HUE DISTRIBUTION for each of the bins
def feature4_23(hsv):
    distribution = [0]*20
    bins = bins(hsv)
    for i in range(len(bins)):
        distribution[i] = bins[i] / imgsize(hsv)
    return distribution
    
def dominant_color(hsv):
    
#COMPOSITION FEATURES START HERE


    ''' 
    PLUS number of quantized hues(number of colors that a painting consists of) 
    hue distribution (19 features)
    find three largest segments for each image(horizontal and vertical coordinates of the mass centers)
    find average saturation for each segment
    find average brightness for each segment
    '''
    

def main():
    f = open('imgurls.txt','w')
    art = open("wikiart.data", "rb")
    labels = open("labels.data", "rb")
    labels_list = list(labels)
    only_url = open("only_url.data", "w")
    only_url_read = open("only_url.data", "r")
    
    #convert_json_to_lists(labels_list)
    genre_id = pickle.load(open('genre_id.p', 'rb'))
    genre_url = pickle.load(open('genre_url.p', 'rb'))
    
    
    
    path = '/Users/noel/Desktop/wikiart/images'

    
    #this wil be used for evaluation 
    #no_labels_ID = nolabels_list(genre_id)
    #print(no_labels_ID)
    
    #pickle.dump(no_labels_ID, open('no_labels.p', 'wb'))
    no_labels_ID=pickle.load(open('no_labels.p', 'rb'))
    #print("HSV")
    
    all_hsv = convert_all_to_HSV(no_labels_ID)
    print(type(all_hsv))
    #all_hsv = np.array(all_hsv, dtype = object)
    #print(all_hsv)
    #all_hsv = np.array(all_hsv)
    print(type(all_hsv))
    
    #h5f = h5py.File('all-hsv.hdf5', 'w')
    #grp = h5f.create_group('alist')
    #h5f.create_dataset('alist', data=all_hsv)
    #h5f.close()
    #all_hsv = np.load('hsv_values1.npy')
    art.close()
    labels.close()
    
    
main()
