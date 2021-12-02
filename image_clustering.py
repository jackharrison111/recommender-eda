# for loading/processing the images  
from keras.preprocessing.image import load_img 
from keras.preprocessing.image import img_to_array 
from keras.applications.vgg16 import preprocess_input 

# models 
from keras.applications.vgg16 import VGG16 
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle

path = r"data/images"


# this list holds all the image filename
flowers = []

# creates a ScandirIterator aliased as files
with os.scandir(path) as files:
  # loops through each file in the directory
    for file in files:
        if file.name.endswith('.jpg'):
          # adds only the image files to the flowers list
            flowers.append(file.name)
            
# change the working directory to the path where the images are located
os.chdir(path)
            
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)

def extract_features(file, model, in_height=224, in_width=224):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(in_height,in_width))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img) 
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1,in_height,in_width,3) 
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features
   
data = {}
p = r"../image_features/features.pkl"

# lop through each image in the dataset
for flower in flowers:
    # try to extract the features and update the dictionary
    try:
        feat = extract_features(flower,model)
        data[flower] = feat
    # if something fails, save the extracted features as a pickle file (optional)
    except:
        print(f"Failed on {flower}")
          
with open(p,'wb') as file:
    pickle.dump(data,file)

# get a list of the filenames
filenames = np.array(list(data.keys()))

# get a list of just the features
feat = np.array(list(data.values()))

# reshape so that there are 210 samples of 4096 vectors
feat = feat.reshape(-1,4096)

'''
# get the unique labels (from the flower_labels.csv)
df = pd.read_csv('flower_labels.csv')
label = df['label'].tolist()
unique_labels = list(set(label))
'''
unique_labels = 8

# reduce the amount of dimensions in the feature vector
pca = PCA(n_components=100, random_state=22)
pca.fit(feat)
x = pca.transform(feat)

# cluster feature vectors
kmeans = KMeans(n_clusters=unique_labels, random_state=22)
kmeans.fit(x)

# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,kmeans.labels_):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)

# function that lets you view a cluster (based on identifier)        
def view_cluster(cluster):
    plt.figure(figsize = (25,25));
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to be shown at a time
    if len(files) > 30:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:29]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1);
        img = load_img(file)
        img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    if save:
        plt.savefig(f'../image_features/cluster_examples-{cluster}.jpg')
    plt.show()
    

save = False 

for cluster in groups.keys():
    view_cluster(cluster) 

#Plot the features using TSNE

from sklearn.manifold import TSNE
 
model_tsne = TSNE(random_state=0)
np.set_printoptions(suppress=True)
 
tsne=model_tsne.fit_transform(x)

max_keys = np.array(list(groups.keys())).max()
fig,ax = plt.subplots(figsize=(10,8))
cmap = plt.get_cmap('gist_rainbow', max_keys)

for vector, key in zip(tsne, kmeans.labels_):

    cax = ax.scatter(vector[0], vector[1], c=key, cmap=cmap, 
                vmin=0, vmax=max_keys)
    
    ax.text(vector[0], vector[1], str(key), color="red", fontsize=12)

plt.title(f'Clustering of images with n_clusters={max_keys}')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
fig.colorbar(cax)#, extend='min')
if save:  
    plt.savefig('../image_features/clustering_predictions_PCAimages_3011.png')
plt.show()


# this is just incase you want to see which value for k might be the best 
sse = []
list_k = list(range(3, 50))

for k in list_k:
    km = KMeans(n_clusters=k, random_state=22)
    km.fit(x)
    
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse)
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance')
if save:
    plt.savefig('../image_features/effect_of_varying_num_clusters_3011.png')
plt.show()
