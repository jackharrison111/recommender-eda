from numpy import unique
from numpy import where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.mixture import GaussianMixture

features = 2

# initialize the data set we'll work with
training_data, _ = make_blobs(
    n_samples=1000,
    n_features=features,
    #n_informative=2,
    #n_redundant=0,
    #n_clusters_per_class=1,
    random_state=4
)

# define the model
gaussian_model = GaussianMixture(n_components=2)

# train the model
gaussian_model.fit(training_data)

# assign each data point to a cluster
gaussian_result = gaussian_model.predict(training_data)

# get all of the unique clusters
gaussian_clusters = unique(gaussian_result)


fig, axes = plt.subplots(1,1, figsize=(8,6))

cmap = plt.cm.get_cmap('hsv', features)

print(gaussian_result)

# plot Gaussian Mixture the clusters
xmin=-5
xmax=5
ymin=-5
ymax=5
for i in range(len(training_data)):
    # get data points that fall in this cluster
    label = gaussian_result[i]
    x = training_data[i][0]
    y = training_data[i][1]
    xmin = x if x < xmin else xmin
    xmax = x if x > xmax else xmax
    ymin = y if y < ymin else ymin
    ymax = y if y > ymax else ymax
    axes.scatter(x, y,  color=cmap(label))

plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
# Fixing the layout to it the size
fig.tight_layout()
# show the Gaussian Mixture plot
#plt.savefig('plots/GMM_5Centroids.jpg')
plt.show()