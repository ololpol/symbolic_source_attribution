import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
import numpy as np
import os


from main import compute_dist, compute_attribution

def verify_plot_folder(subfolder = None):
    if not os.path.exists("plots"):
        os.makedirs("plots", exist_ok=True)

    if not os.path.exists("plots/embeddings"):
        os.makedirs("plots/embeddings", exist_ok=True)
    if not os.path.exists("plots/distances"):
        os.makedirs("plots/distances", exist_ok=True)
    if not os.path.exists("plots/attribution"):
        os.makedirs("plots/attribution", exist_ok=True)
    if subfolder is not None:
        os.makedirs("plots/attribution/"+subfolder, exist_ok=True)
    #TODO check if a set of subfolders exist in the data folder, and create them if they don't
def extract_embeddings(data, labels, embedding):
    # Extract the embeddings for the specified labels and embedding type
    extracted = None
    for label in labels:
        if extracted is None:
            extracted = data[label]["embed"][embedding]
        else: 
            extracted = np.vstack([extracted, data[label]["embed"][embedding]])
    return extracted

def plot_embeddings(data, labels = "ONeill", embedding = "clamp", name=None):
    """
    Generates a 2D scatter plots of the first two dimensions of the embeddings and saves it as a PNG file.
        
    Args:
        data (dict): A dictionary containing the data to be plotted.
        labels (str or list[str]): The label(s) of the data to be plotted.
        embedding (str): The embedding type to be plotted.
        name (str): A string used in the title and filename of the plot.
    Returns:
        None
            
    Image is saved as embeddings_{name}.png, default name is the label and embedding type.
    """
    if isinstance(labels, str):
        labels = [labels]
    embeddings = extract_embeddings(data, labels, embedding)
    if name is None:
        label_str = "-".join(labels)
        name = f"{label_str}_{embedding}"


    plt.clf()
    plt.figure(figsize=(8, 6))
    xx = [e[0] for e in embeddings]
    yy = [e[1] for e in embeddings]
    plt.scatter(xx, yy, alpha=0.5)
    plt.title(f"Embeddings for {name}")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid()
    plt.savefig(f"plots/embeddings/{name}.png")


def plot_embeddings_pca(data, labels = "ONeill", embedding = "clamp", name=None, components=5):
    """
    Generates a 2D scatter plots of the first two dimension of the PCA-reduced embeddings and saves it as a PNG file.
        
    Args:
        data (dict): A dictionary containing the data to be plotted.
        labels (str or list[str]): The label(s) of the data to be plotted.
        embedding (str): The embedding type to be plotted.
        name (str): A string used in the title and filename of the plot.
    Returns:
        None
            
    Image is saved as embeddings_pca_{name}.png, default name is the label(s) and embedding type.
    """
    if isinstance(labels, str):
        labels = [labels]
    embeddings = extract_embeddings(data, labels, embedding)
    if name is None:
        label_str = "-".join(labels)
        name = f"{label_str}_{embedding}"


    pca = PCA(n_components=components)
    reduced_embeddings = pca.fit_transform(embeddings)

    #print("PCA VR:", pca.explained_variance_ratio_)


    plt.clf()
    plt.figure(figsize=(8, 6))
    xx = [e[0] for e in reduced_embeddings]
    yy = [e[1] for e in reduced_embeddings]
    plt.scatter(xx, yy, alpha=0.5)
    plt.title(f"PCA-reduced Embeddings for {name}")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.savefig(f"plots/embeddings/pca_{name}.png")

def plot_pca_variance(data, labels = "ONeill", embedding = "clamp", name=None, components=5):
    """
    Generates a plot of the explained variance ratio of the first PCA components and saves it as a PNG file.
        
    Args:
        data (dict): A dictionary containing the data to be plotted.
        labels (str or list[str]): The label(s) of the data to be plotted.
        embedding (str): The embedding type to be plotted.
        name (str): A string used in the title and filename of the plot.
    Returns:
        None
            
    Image is saved as pca_variance_{name}.png, default name is the label(s) and embedding type.
    """
    if isinstance(labels, str):
        labels = [labels]
    embeddings = extract_embeddings(data, labels, embedding)
    if name is None:
        label_str = "-".join(labels)
        name = f"{label_str}_{embedding}"


    pca = PCA(n_components=components)
    pca.fit(embeddings)

    acc = 0
    var_sums = []
    for var in pca.explained_variance_ratio_:
        acc += var
        var_sums.append(acc)
    
    
    plt.clf()
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, components+1), var_sums, marker='o')
    plt.title(f"Cumulative Explained Variance Ratio")
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid()
    plt.savefig(f"plots/embeddings/pca_variance_{name}.png")

def plot_pca_pairs(data, labels = "ONeill", embedding = "clamp", name=None):
    """
    Generates a grid of scatter plots for all pairs of the first 5 principal components and saves it as a PNG file.
        
    Args:
        data (dict): A dictionary containing the data to be plotted.
        labels (str or list[str]): The label(s) of the data to be plotted.
        embedding (str): The embedding type to be plotted.
        name (str): A string used in the title and filename of the plot.
    Returns:
        None
            
    Image is saved as pca_pairs_{name}.png, default name is the label(s) and embedding type.
    """
    if isinstance(labels, str):
        labels = [labels]
    embeddings = extract_embeddings(data, labels, embedding)
    if name is None:
        label_str = "-".join(labels)
        name = f"{label_str}_{embedding}"


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
    plt.savefig(f"plots/embeddings/pca_pairs_{name}.png")


def plot_distance_distribution(dists, name, t_step=0.05):

    t_steps = math.ceil(1/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = [sum([1 for d in dists if d > t and d <= t+t_step]) for t in thresholds]

    plt.clf()
    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of distances")
    plt.title("Distribution of distances to output embedding")
    plt.savefig(f"plots/distances/distribution_{name}.png")


def plot_origin_distance(data, labels = "ONeill", embedding = "clamp", method="euclidean", name=None, t_step=0.2):

    if isinstance(labels, str):
        labels = [labels]
    embeddings = extract_embeddings(data, labels, embedding)

    if name is None:
        label_str = "-".join(labels)
        name = f"{label_str}_{embedding}_{method}"

    embed_len = len(embeddings[0])
    origin = [0]*embed_len

    dists = compute_dist(origin, embeddings, method=method)[0]
    dist_max = max(dists)

    t_steps = math.ceil(dist_max/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = [sum([1 for d in dists if d > t and d <= t+t_step]) for t in thresholds]

    
    plt.clf()
    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of distances")
    plt.title("Distribution of distances to origin")
    plt.savefig(f"plots/distances/origin_dist_{name}.png")


def plot_centroid_distance(data, labels = "ONeill", embedding = "clamp", method="euclidean", name=None, t_step=0.2):

    if isinstance(labels, str):
        labels = [labels]
    embeddings = extract_embeddings(data, labels, embedding)

    if name is None:
        label_str = "-".join(labels)
        name = f"{label_str}_{embedding}_{method}"

    embed_len = len(embeddings[0])


    centroid = [sum([e[i] for e in embeddings])/len(embeddings) for i in range(embed_len)]
    
    dists = compute_dist(centroid, embeddings, method=method)[0]
    dist_max = max(dists)

    t_steps = math.ceil(dist_max/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = [sum([1 for d in dists if d > t and d <= t+t_step]) for t in thresholds]

    
    plt.clf()
    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of distances")
    plt.title("Distribution of distances to centroid")
    plt.savefig(f"plots/distances/centroid_dist_{name}.png")

def plot_distance_distribution(dists, name, t_step=0.05):

    t_steps = math.ceil(1/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = [sum([1 for d in dists if d > t and d <= t+t_step]) for t in thresholds]

    plt.clf()
    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of distances")
    plt.title("Distribution of distances to output embedding")
    plt.savefig(f"plots/distances/distribution_{name}.png")

def plot_distance_average(data, source_labels = "ONeill", target_labels = None, embedding = "clamp", method="cosine", name = None, t_step=0.05):
    """
    Generates a 2D scatter plots of the first two dimensions of the embeddings and saves it as a PNG file.
        
    Args:
        data (dict): A dictionary containing the data to be plotted.
        source_labels (str or list[str]): the labels of the data subset to compute distances from.
        target_labels (str or list[str]): the labels of the data subset to compute distances to.
        embedding (str): The embedding type to use as data.
        method (str): The method to use for distance computation.
        name (str): A string used in the title and filename of the plot.
        t_step (float): The step size for distance thresholds in the plot.
    Returns:
        None
            
    Image is saved as distance_average_{name}.png", default name is the source labels + target labels + embedding + method.
    """
     

    if isinstance(source_labels, str):
        source_labels = [source_labels]
    if target_labels is None:
        target_labels = source_labels
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    source_embeddings = extract_embeddings(data, source_labels, embedding)
    target_embeddings = extract_embeddings(data, target_labels, embedding)

    if name is None:
        source_label_str = "-".join(source_labels)
        target_label_str = "-".join(target_labels)
        if source_label_str == target_label_str:
             name = f"{source_label_str}_{embedding}"
        else: 
            name = f"{source_label_str}_{target_label_str}_{embedding}_{method}"


    #print("Computing distances between", source_labels, "and", target_labels, "using", embedding)

    dists = compute_dist(source_embeddings, target_embeddings, method=method)
    #for e in source_embeddings:
    #    out_dists = compute_dist(e, target_embeddings, method=method)
    #    dists.append(out_dists)

    N = len(dists) # = len(source_embeddings)
    
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
    plt.ylabel("Average number of distances within threshold")
    plt.title("Average distribution of distances")
    plt.savefig(f"plots/distances/average_{name}.png")


def avg_distance_bars(data, source_labels = "ONeill", target_labels = None, embedding = "clamp", method="cosine", name = None):

    #print("avg distancse bar for ", source_labels, "to", target_labels, "using", embedding)
    #TODO description
    if isinstance(source_labels, str):
        source_labels = [source_labels]
    if target_labels is None:
        target_labels = source_labels
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    source_embeddings = extract_embeddings(data, source_labels, embedding)

    if name is None:
        source_label_str = "-".join(source_labels)
        name = f"{source_label_str}_{embedding}_{method}"

    res = [0]*len(target_labels)
    
    for i, label in enumerate(target_labels):
        target_embeddings = data[label]["embed"][embedding]
        #print("target embeddings:", target_embeddings)
        #print("shape: ", target_embeddings.shape)
        #print("source shape: ", source_embeddings.shape)
        n_items = len(source_embeddings)
        dists = [0]*len(source_embeddings)
        for e in source_embeddings:
            out_dists = compute_dist(e, target_embeddings, method=method)[0]
            #TODO this can be optimized by computing the distance between all source and target embeddings at once
            res[i] += np.average(out_dists)
        res[i] /= n_items

    #print("Distance stats: min {}, max {}, mean {}, median {}".format(np.min(dists_flat), np.max(dists_flat), np.mean(dists_flat), np.median(dists_flat)))

    plt.clf()
    plt.bar(target_labels, res)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average distance")
    plt.title(f"Avg distance for {name}")
    plt.tight_layout()
    plt.savefig(f"plots/distances/avg_dist_{name}.png")

def plot_min_pos(extracted, id_map, label_map, name = None):

    unused_labels =  set(label_map.keys())
    min_pos = [0] * len(unused_labels)
    for i, item in enumerate(extracted):
        label, pos, dist = item
        if label in unused_labels:
            id = label_map[label]
            min_pos[id] = i
            unused_labels.remove(label)

    plt.clf()
    plt.bar(id_map.values(), min_pos)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Minimum position")
    plt.title(f"Minimum position for {name}")
    plt.tight_layout()
    plt.savefig(f"plots/distances/min_pos_{name}.png")
         


def plot_attribution(attribution, id_map, name, config_label = None):
    #TODO maybe make this work directly on data dict?

    plt.clf()
    plt.bar(id_map.values(), attribution)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Attribution value")
    plt.title(f"Attribution for {name}")
    plt.tight_layout()
    if config_label is not None:
        plt.savefig(f"plots/attribution/{config_label}/{name}.png")
    plt.savefig(f"plots/attribution/{name}.png")

def plot_attribution_distribution(data, source_labels = "ONeill", target_labels = None, embedding = "clamp", method="cosine", name=None, top_N = None, dist_threshold = None, top_Y = None, attribution_threshold = None, config_label = None):
    

    if isinstance(source_labels, str):
        source_labels = [source_labels]

    if target_labels == None:
        target_labels = source_labels
    if isinstance(target_labels, str):
        target_labels = [target_labels]


    if name is None:
        source_str = "-".join(source_labels)
        name = f"{source_str}_{embedding}_{method}"

    attribution, id_map, _ = compute_attribution(data, source_labels, target_labels, embedding=embedding, top_N=top_N, dist_threshold=dist_threshold, top_Y=top_Y, attribution_threshold=attribution_threshold)

    N_artists = len(attribution[0])
    counts = [0]*N_artists
    avg = [0]*N_artists

    #TODO could probably do some numpy stuff here
    for i in range(len(attribution)):
        amax = np.argmax(attribution[i])
        counts[amax] += 1
        for j, e in enumerate(attribution[i]):
            avg[j] += e

    for i in range(len(avg)):
        avg[i] = avg[i] / len(attribution)

    plt.clf()
    plt.bar(id_map.values(), counts)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Number of times attributed")
    plt.title(f"Attribution distribution for {name}")
    plt.tight_layout()
    if config_label is not None:
        plt.savefig(f"plots/attribution/{config_label}/distribution_{name}.png")
    plt.savefig(f"plots/attribution/distribution_{name}.png")

    plt.clf()
    plt.bar(id_map.values(), avg)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Average Attribution")
    plt.title(f"Average attribution for {name}")
    plt.tight_layout()
    if config_label is not None:
        plt.savefig(f"plots/attribution/{config_label}/average_{name}.png")
    plt.savefig(f"plots/attribution/average_{name}.png")