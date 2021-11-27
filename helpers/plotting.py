import matplotlib.pyplot as plt


def plot_umap(fit_data, targets, rand_state=None, title=None, colour="gist_rainbow",
              save_name=None):

    import umap
    
    fit = umap.UMAP(random_state=rand_state)
    u = fit.fit_transform(fit_data)
    fig,ax = plt.subplots(figsize=(10,8))
    cmap = plt.get_cmap('gist_rainbow', 10)
    cax = ax.scatter(u[:,0], u[:,1], s=5, c=targets, cmap=cmap, 
                 vmin=0, vmax=targets.max())
    fig.colorbar(cax)#, extend='min')
    plt.title(title)
    if save_name != None:
        print("Saving...")
        plt.savefig(fname=f"{save_name}", quality=100)
    plt.show()


def plot_clusters()