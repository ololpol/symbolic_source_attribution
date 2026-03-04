import matplotlib.pyplot as plt
import math


def plot_embeddings(embeddings, name):
    #TODO use dimensionality reduction instead here
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