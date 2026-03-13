import matplotlib.pyplot as plt
import math

from main import compute_dist


def plot_embeddings(embeddings, name):
    plt.clf()
    plt.figure(figsize=(8, 6))
    xx = [e[0] for e in embeddings]
    yy = [e[1] for e in embeddings]
    plt.scatter(xx, yy, alpha=0.5)
    plt.title(f"Embeddings for {name}")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid()
    plt.savefig(f"plots/embeddings_{name}.png")


def plot_embeddings_pca(embeddings, name):

    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    reduced_embeddings = pca.fit_transform(embeddings)

    print("PCA VR:", pca.explained_variance_ratio_)


    plt.clf()
    plt.figure(figsize=(8, 6))
    xx = [e[0] for e in reduced_embeddings]
    yy = [e[1] for e in reduced_embeddings]
    plt.scatter(xx, yy, alpha=0.5)
    plt.title(f"PCA-reduced Embeddings for {name}")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.savefig(f"plots/embeddings_pca_{name}.png")

def plot_pca_variance(embeddings, name):
    #TODO UHHH this is defenitly wrong
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    pca.fit(embeddings)

    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, 6), pca.explained_variance_ratio_)
    plt.title(f"PCA Explained Variance Ratio for {name}")
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.grid()
    plt.savefig(f"plots/pca_variance_{name}.png")

def plot_pca_pairs(embeddings, name):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.clf()
    plt.figure(figsize=(12, 10))
    for i in range(5):
        for j in range(i+1, 5):
            plt.subplot(5, 5, i*5 + j)
            xx = [e[i] for e in reduced_embeddings]
            yy = [e[j] for e in reduced_embeddings]
            plt.scatter(xx, yy, alpha=0.5)
            plt.xlabel(f'PC {i+1}')
            plt.ylabel(f'PC {j+1}')
            plt.grid()
    plt.suptitle(f"PCA Pairwise Plots for {name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/pca_pairs_{name}.png")


def plot_distance_distribution(dists, name, t_step=0.05):
    # Plot dists_sorted as a bar chart, x axis is distance threshould, y axis is nr of distances
    
    t_steps = math.ceil(1/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = [sum([1 for d in dists if d > t and d <= t+t_step]) for t in thresholds]

    plt.clf()
    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of distances")
    plt.title("Distribution of distances to output embedding")
    plt.savefig(f"plots/distance_distribution_{name}.png")


def plot_distance_average(embedded_data, name, method = "cosine", t_step=0.05):
    # Plot the average distance to the output embedding for each distance threshold

    dists = []
    for e in embedded_data:
        out_dists = compute_dist(e, embedded_data, method=method)
        dists.append(out_dists[method])

    N = len(embedded_data)

    t_steps = math.ceil(1/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = []
    for d in dists:
        dist_counts_d = [sum([1 for de in d if de > t and de <= t+t_step]) for t in thresholds]
        for i in range(len(dist_counts_d)):
            if len(dist_counts) <= i:
                dist_counts.append(dist_counts_d[i])
            else:
                dist_counts[i] += dist_counts_d[i]

    for i in range(len(dist_counts)):
        dist_counts[i] /= N

    plt.clf()
    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Average distance between embeddings")
    plt.title("Average distance between embeddings by distance threshold")
    plt.savefig(f"plots/distance_average_{name}_{method}.png")