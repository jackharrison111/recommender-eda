from numpy import unique
from numpy import where
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture

features = 5

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=features,
    n_informative=5,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the model
gaussian_model = GaussianMixture(n_components=5)

# train the model
gaussian_model.fit(training_data)

# assign each data point to a cluster
gaussian_result = gaussian_model.predict(training_data)

# get all of the unique clusters
gaussian_clusters = unique(gaussian_result)


fig, axes = plt.subplots(1,1, figsize=(8,6))

cmap = plt.cm.get_cmap('hsv', features)

print(gaussian_clusters)

# plot Gaussian Mixture the clusters

for i in range(len(training_data)):
    # get data points that fall in this cluster
    label = gaussian_result[i]
    axes.scatter(training_data[i][0], training_data[i][1], color=cmap(label))

plt.xlim([-5, 5])
plt.ylim([-5,5])
# Fixing the layout to it the size
fig.tight_layout()
# show the Gaussian Mixture plot
#plt.savefig('plots/GMM_5Centroids.jpg')
plt.show()