import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
import numpy as np
import os
import random

#import from main if not already imported, to avoid circular imports
if "compute_dist" not in globals():
    from main import compute_dist
if "compute_attribution" not in globals():
     from main import compute_attribution




def verify_plot_folder(subfolder = None):
    if not os.path.exists("plots"):
        os.makedirs("plots", exist_ok=True)

    if not os.path.exists("plots/data"):
        os.makedirs("plots/data", exist_ok=True)
    if not os.path.exists("plots/embeddings"):
        os.makedirs("plots/embeddings", exist_ok=True)
    if not os.path.exists("plots/distances"):
        os.makedirs("plots/distances", exist_ok=True)
    if not os.path.exists("plots/attribution"):
        os.makedirs("plots/attribution", exist_ok=True)
    if subfolder is not None:
        os.makedirs("plots/attribution/"+subfolder, exist_ok=True)
    #TODO check if a set of sub
    # 
    # folders exist in the data folder, and create them if they don't

def plot_data_dist(data, labels, name):
    print(data.keys())
    res = [0] * len(labels)
    for i, label in enumerate(labels):
        label_dict = data[label]
        embedding = list(data[label].keys())[0]
        res[i] = len(data[label][embedding])


    plt.clf()
    plt.bar(labels, res)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right", fontsize=16)
    plt.ylabel("Number of items")
    #plt.title(f"Data distribution for {name}")
    plt.tight_layout()
    plt.savefig(f"plots/data/distribution_{name}.png")

def extract_embeddings(data, labels, embedding):
    # Extract the embeddings for the specified labels and embedding type
    extracted = None
    for label in labels:
        if extracted is None:
            extracted = data[label][embedding]
        else: 
            extracted = np.vstack([extracted, data[label][embedding]])
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
    #plt.title(f"Embeddings for {name}")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid()
    plt.savefig(f"plots/embeddings/{name}.png")
    plt.close()



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
    #plt.title(f"PCA-reduced Embeddings for {name}")
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.savefig(f"plots/embeddings/pca_{name}.png")
    plt.close()


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
    #print("Embeddings shape:", embeddings.shape)
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
    #plt.title(f"Cumulative Explained Variance Ratio")
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.grid()
    plt.savefig(f"plots/embeddings/pca_variance_{name}.png")
    plt.close()


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
    #plt.suptitle(f"PCA Pairwise Plots for {name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plots/embeddings/pca_pairs_{name}.png")
    plt.close()



def plot_distance_distribution(dists, name, t_step=0.05):

    t_steps = math.ceil(1/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = [sum([1 for d in dists if d > t and d <= t+t_step]) for t in thresholds]

    plt.clf()
    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of distances")
    #plt.title("Distribution of distances to output embedding")
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
    #plt.title("Distribution of distances to origin")
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
    #plt.title("Distribution of distances to centroid")
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
    #plt.title("Distribution of distances to output embedding")
    plt.savefig(f"plots/distances/distribution_{name}.png")


def get_shortest_dists(data, label = "ONeill", embedding = "clamp", method ="cosine", name = None, N=10):
    if isinstance(label, str):
        label = [label]
    embeddings = extract_embeddings(data, label, embedding)

    if name is None:
        label_str = "-".join(label)
        name = f"{label_str}_{embedding}_{method}"

    dists = compute_dist(embeddings, embeddings, method=method)
    for i in range(len(dists)):
        dists[i][i] = float("inf") # ignore self distance
    
    #TODO this can be optimized by using numpy functions instead of flattening and sorting
    #TODO this breaks if N is big and we get the self distances
    order = np.argsort(dists.flatten())
    shortest_dists = [dists.flatten()[i] for i in order[:N]]
    indices = [(i//len(dists), i%len(dists)) for i in order[:N]]

    with open(f"plots/distances/shortest_dists_{name}.txt", "w") as f:
        for i in range(N):
            f.write(f"{shortest_dists[i]}: {indices[i]}\n")

    

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
             name = f"self_{source_label_str}_{embedding}"
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
    plt.ylabel("Average number of distances")
    #plt.title("Average distribution of distances")
    plt.savefig(f"plots/distances/average_{name}.png")

def local_dist_grid(data, label, embedding ="clamp", method = "cosine", name = None):
    
    embeddings = data[label][embedding]
    dists = compute_dist(embeddings, embeddings, method=method)

    plt.clf()
    plt.imshow(dists, cmap='viridis', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Distance')
    #plt.title(f"Distance grid for {label} using {embedding}")
    plt.xlabel("Item index")
    plt.ylabel("Item index")
    plt.savefig(f"plots/distances/grid_{label}_{embedding}_{method}.png")

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
        target_embeddings = data[label][embedding]
        #print("target embeddings:", target_embeddings)
        #print("shape: ", target_embeddings.shape)
        #print("source shape: ", source_embeddings.shape)
        n_items = len(source_embeddings)
        dists = [0]*len(source_embeddings)
        for e in source_embeddings:
            out_dists = compute_dist([e], target_embeddings, method=method)[0]
            #TODO this can be optimized by computing the distance between all source and target embeddings at once
            res[i] += np.average(out_dists)
        res[i] /= n_items

    #print("Distance stats: min {}, max {}, mean {}, median {}".format(np.min(dists_flat), np.max(dists_flat), np.mean(dists_flat), np.median(dists_flat)))

    plt.clf()
    plt.bar(target_labels, res)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right", fontsize=16)
    plt.ylabel("Average distance")
    #plt.title(f"Avg distance for {name}")
    plt.tight_layout()
    plt.savefig(f"plots/distances/avg_dist_{name}.png")

def plot_min_pos(extracted, id_map, label_map, name = None):

    unused_labels = set(label_map.keys())
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
    plt.xticks(rotation=30, ha="right", fontsize=16)
    plt.ylabel("Minimum position")
    #plt.title(f"Minimum position for {name}")
    plt.tight_layout()
    plt.savefig(f"plots/distances/min_pos_{name}.png")

def avg_min_pos(extracted, id_map, label_map, name = None):

    
    min_pos = [0] * len(label_map)

    for i, items in enumerate(extracted):
        unused_labels = set(label_map.keys())
        for j, item in enumerate(items):
            label, pos, dist = item
            if label in unused_labels:
                id = label_map[label]
                min_pos[id] += j
                unused_labels.remove(label)

    for i in range(len(min_pos)):
        min_pos[i] /= len(extracted)

    plt.clf()
    plt.bar(id_map.values(), min_pos)
    plt.xlabel("Artist")
    plt.xticks(rotation=30, ha="right", fontsize=16)
    plt.ylabel("Average minimum position")
    #plt.title(f"Average minimum position for {name}")
    plt.tight_layout()
    plt.savefig(f"plots/distances/avg_min_pos_{name}.png")
         


def plot_attribution(attribution, id_map, name, config_label = None):
    #TODO maybe make this work directly on data dict?

    plt.clf()
    
    # Check if id_map values contain key-meter pairs (separated by "-")
    id_values = list(id_map.values())
    if id_values and "-" in str(id_values[0]):
        # Parse key-meter pairs and create a 2D grid plot
        key_meter_pairs = {}
        for idx, label in id_map.items():
            key_meter_pairs[idx] = label.split("-")
        
        # Extract unique keys and meters, preserving order
        all_keys = []
        all_meters = []
        for idx in sorted(key_meter_pairs.keys()):
            key, meter = key_meter_pairs[idx]
            if key not in all_keys:
                all_keys.append(key)
            if meter not in all_meters:
                all_meters.append(meter)
        
        # Create 2D grid for attribution values
        grid = np.zeros((len(all_keys), len(all_meters)))
        key_to_idx = {key: i for i, key in enumerate(all_keys)}
        meter_to_idx = {meter: i for i, meter in enumerate(all_meters)}
        
        for idx, att_val in enumerate(attribution):
            if idx in key_meter_pairs:
                key, meter = key_meter_pairs[idx]
                grid[key_to_idx[key], meter_to_idx[meter]] = att_val
        
        # Create 2D grid plot
        fig, ax = plt.subplots(figsize=(max(10, len(all_meters) * 0.8), max(8, len(all_keys) * 0.8)))
        
        im = ax.imshow(grid, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(all_meters)))
        ax.set_yticks(range(len(all_keys)))
        ax.set_xticklabels(all_meters, rotation=45, ha='right', fontsize=20)
        ax.set_yticklabels(all_keys, fontsize=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attribution value', rotation=270, labelpad=15)
        
        # Add text annotations for values
        #for i, key in enumerate(all_keys):
        #    for j, meter in enumerate(all_meters):
        #        value = grid[i, j]
        #        if value > 0:
        #            text = ax.text(j, i, f'{value:.2f}', ha="center", va="center",
        #                         color="black" if value < grid.max() / 2 else "white",
        #                         fontsize=9)
        
        #ax.set_xlabel('Key', fontsize=12)
        #ax.set_ylabel('Meter', fontsize=12)
        #ax.set_title(f"Attribution for {name}")
        
        plt.tight_layout()
        if config_label != None:
            plt.savefig(f"plots/attribution/{config_label}/grid_{name}.png", dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f"plots/attribution/grid_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()
    else:
        # Original bar chart for non-grid data
        plt.bar(id_map.values(), attribution)
        plt.xlabel("Artist")
        plt.xticks(rotation=30, ha="right", fontsize=16)
        plt.ylabel("Attribution value")
        #plt.title(f"Attribution for {name}")
        plt.tight_layout()
        if config_label != None:
            plt.savefig(f"plots/attribution/{config_label}/{name}.png")
        else:
            plt.savefig(f"plots/attribution/{name}.png")
        plt.close()

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


    id_values = list(id_map.values())
    if id_values and "-" in str(list(id_map.values())[0]):
        # Parse key-meter pairs and create a 2D grid plot
        key_meter_pairs = {}
        for idx, label in id_map.items():
            key_meter_pairs[idx] = label.split("-")
        
        # Extract unique keys and meters, preserving order
        all_keys = []
        all_meters = []
        for idx in sorted(key_meter_pairs.keys()):
            key, meter = key_meter_pairs[idx]
            if key not in all_keys:
                all_keys.append(key)
            if meter not in all_meters:
                all_meters.append(meter)


        target_vectors = [counts, avg]
        names = ["distribution", "average"]


        for id in range(len(target_vectors)):

            # Create 2D grid for attribution values
            grid = np.zeros((len(all_keys), len(all_meters)))
            key_to_idx = {key: i for i, key in enumerate(all_keys)}
            meter_to_idx = {meter: i for i, meter in enumerate(all_meters)}
            
            for idx, att_val in enumerate(target_vectors[id]):
                if idx in key_meter_pairs:
                    key, meter = key_meter_pairs[idx]
                    grid[key_to_idx[key], meter_to_idx[meter]] = att_val
            if names[id] == "average":
                f = open(f"plots/attribution/{config_label}/dump_{name}.txt", "w")
                f.write(str(grid))
                f.close()
            # Create 2D grid plot
            fig, ax = plt.subplots(figsize=(max(10, len(all_meters) * 0.8), max(8, len(all_keys) * 0.8)))
            
            im = ax.imshow(grid, cmap='viridis', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks(range(len(all_meters)))
            ax.set_yticks(range(len(all_keys)))
            ax.set_xticklabels(all_meters, rotation=45, ha='right', fontsize=20)
            ax.set_yticklabels(all_keys, fontsize=20)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if id == 0:
                cbar.set_label('Attribution count', rotation=270, labelpad=15)
            else:
                cbar.set_label('Attribution average', rotation=270, labelpad=15)
            
            # Add text annotations for values
            #for i, key in enumerate(all_keys):
            #    for j, meter in enumerate(all_meters):
            #        value = grid[i, j]
            #        if value > 0:
            #            text = ax.text(j, i, f'{value:.2f}', ha="center", va="center",
            #                        color="black" if value < grid.max() / 2 else "white",
            #                        fontsize=9)
            
            #ax.set_xlabel('Key', fontsize=12)
            #ax.set_ylabel('Meter', fontsize=12)

            #if id == 0:
            #    ax.set_title(f"Attribution counts {name}")
            #else:
            #    ax.set_title(f"Average attribution for {name}")
            
            plt.tight_layout()
            if config_label != None:
                plt.savefig(f"plots/attribution/{config_label}/{names[id]}_{name}_grid.png", dpi=150, bbox_inches='tight')
            else:
                plt.savefig(f"plots/attribution/{names[id]}_{name}_grid.png", dpi=150, bbox_inches='tight')
            plt.close()



    else: 
        plt.clf()
        plt.bar(id_map.values(), counts)
        plt.xlabel("Artist")
        plt.xticks(rotation=30, ha="right", fontsize=16)
        plt.ylabel("Number of times attributed")
        #plt.title(f"Attribution distribution for {name}")
        plt.tight_layout()
        if config_label != None:
            plt.savefig(f"plots/attribution/{config_label}/distribution_{name}.png")
        else:
            plt.savefig(f"plots/attribution/distribution_{name}.png")

        plt.clf()
        plt.bar(id_map.values(), avg)
        plt.xlabel("Artist")
        plt.xticks(rotation=30, ha="right", fontsize=16)
        plt.ylabel("Average Attribution")
        #plt.title(f"Average attribution for {name}")
        plt.tight_layout()
        if config_label != None:
            plt.savefig(f"plots/attribution/{config_label}/average_{name}.png")
        else:
            plt.savefig(f"plots/attribution/average_{name}.png")
        plt.close()



def make_embedding_plots(data, labels, embeddings):

    print("Making embedding plots for", labels, "using", embeddings)

    for embedding in embeddings:
        for label in labels:
                    # Embedding plots

            print(len(data[label][embedding]), data[label][embedding][0].shape)  

            plot_embeddings(data, label, embedding)



            n_components = min(5, len(data[label][embedding][0]), len(data[label][embedding]))
            plot_embeddings_pca(data, label, embedding, components=n_components)

            if n_components >= 5:
                plot_pca_pairs(data, label, embedding)


            #TODO fix this maybe
            #n_components = 20#min(101, len(data[label][embedding][0]), len(data[label][embedding])) - 1
            #print(len(data[label][embedding]), data[label][embedding][0].shape)  
            #if len(data[label][embedding]) > n_components:
            #    plot_pca_variance(data, label, embedding, components=n_components)


def make_distance_plots(data, source_labels, target_labels = None, embeddings = ["clamp"], methods = ["cosine"]):
    if target_labels is None:
        target_labels = source_labels

    print("Making distance plots for", source_labels, "and", target_labels, "using", embeddings, "and methods", methods)

    for embedding in embeddings:
        # Distance plots
        for m in methods:
            for label1 in source_labels:
                plot_distance_average(data, label1, label1, embedding, method=m)
                get_shortest_dists(data, label1, embedding, method=m, N=20)
                for label2 in target_labels:
                    plot_distance_average(data, label1, label2, embedding, method=m)

            for label in target_labels:
                plot_distance_average(data,label, label, embedding)
            




        for label in source_labels + target_labels:
            plot_origin_distance(data, label, embedding, method="euclidean", t_step=0.01)
            plot_centroid_distance(data, label, embedding, method="euclidean", t_step=0.01)

        #e_out = random.choice(data["ONeill_100"][embedding])
        #for method in methods:
        #    #print("e_out:", e_out)
        #    #print("e_out shape:", e_out.shape)
        #    out_dists = compute_dist(e_out, data["ONeill"][embedding], method)[0]
        #    #print("OD:", out_dists, out_dists.shape) 
        #    out_dists_full = compute_dist(e_out, np.vstack([data["ONeill"][embedding], data["ONeill_100"][embedding]]), method)[0]


        #    plot_distance_distribution(out_dists, embedding+"_"+m, t_step=0.01)
        #    plot_distance_distribution(out_dists_full, embedding+"_full_"+m, t_step=0.01)


        for label in source_labels:
            target_labels = target_labels.copy()
            attribution, id_map, label_map, extracted = compute_attribution(data, source_labels=[label], target_labels = target_labels, attribution_method = None, dist_method = "cosine", embedding = embedding, extract_N=999, extract_write=False)
            #print("extracted:", extracted)
            e = random.randint(0, len(extracted)-1)
            name = label + "_" + embedding + "_item_" + str(e)
            plot_min_pos(extracted[e], id_map, label_map, name)
            avg_min_pos(extracted, id_map, label_map, name=label)
            #TODO average min pos plot?

def make_attribution_plots(data, source_labels, target_labels = None, embeddings = ["clamp"], methods = ["cosine"], configs = []):

    if target_labels is None:
        target_labels = source_labels

    print("Making attribution plots for", source_labels, "and", target_labels, "using", embeddings, "and methods", methods)
    for embedding in embeddings: 
        # Attribution plots
    
        
        for config in configs:
            config_label = ""
            if config[0] != None:
                config_label += "N="+str(config[0])+"_"
            if config[1] != None:
                config_label += "D_T="+str(config[1])+"_"
            if config[2] != None:
                config_label += "Y="+str(config[2])+"_"
            if config[3] != None:
                config_label += "A_T="+str(config[3])+"_"
            if config_label == "":
                config_label = "none"
            verify_plot_folder(subfolder = config_label)
            a = [1,2,3]
            for label in source_labels:
                target_labels_copy = target_labels.copy()
                if label in target_labels:
                    target_labels_copy.remove(label)
                attribution, id_map, label_map = compute_attribution(data, source_labels=[label], target_labels = target_labels_copy, attribution_method = None, dist_method = methods[0], embedding = embedding, top_N = config[0], dist_threshold=config[1], top_Y=config[2], attribution_threshold=config[3])

                #avg_distance_bars(data, [label], target_labels_copy, embedding)
                
                plot_attribution(random.choice(attribution), id_map, label + "_" + embedding, config_label = config_label)
                plot_attribution_distribution(data, [label], target_labels, embedding, top_N = config[0], dist_threshold=config[1], top_Y=config[2], attribution_threshold=config[3], config_label = config_label)


def get_most_similar(data, source_label, target_labels, label_pos, embedding = "clamp", method = "cosine", N=5, out_file = None, dense = False):
    item = data[source_label][embedding][label_pos]


    extracted = None
    id_map = {}
    pos_map = {}
    ids = []
    pos = 0
    i = 0

    out = ""
    for label in target_labels:
        if extracted is None:
            extracted = data[label][embedding]
            
        else: 
            extracted = np.vstack([extracted, data[label][embedding]])

        ids += [i for _ in range(len(data[label][embedding]))]
        id_map[i] = label
        pos_map[label] = pos
        i += 1
        pos += len(data[label][embedding])

    target_data = extracted
    dists = compute_dist(item, target_data, method=method)[0]
    sorted_indices = np.argsort(dists)

    if len(data[source_label][embedding]) <= label_pos:
        print(f"Label position {label_pos} is out of bounds for {source_label}, defaulting to 1st item")
        label_pos = 0
    if not dense:
        out += f"Most similar items to {source_label} item {label_pos} using {embedding} and {method}:\n"
        out += f"Original tune: \n{data[source_label]['abc'][label_pos]}\n"
    else: 
        out += data[source_label]['abc'][label_pos] + "\n\n"

    for i in range(N):
        idx = sorted_indices[i]
        
        target_label = id_map[ids[idx]]
        target_pos = idx - pos_map[target_label]
        if not dense:
            out += f"Most similar {i+1}: {target_label} item {target_pos} (distance: {dists[sorted_indices[i]]})\n"
            out += f"Content: \n{data[target_label]['abc'][target_pos]}\n"
        else:
            out += data[target_label]['abc'][target_pos] + "\n\n"

    if out_file:
        with open(out_file, "w") as f:
            f.write(out)
    else:
        print(out)  


def plot_key_meter_grid(data_dir: str = "data/data_v2", output_name: str = "key_meter_grid"):
    """
    Generate a grid plot showing the sizes of different key and meter pair groups.
    
    Args:
        data_dir (str): Path to the data directory containing music files
        output_name (str): Name for the output plot file (without extension)
    
    Returns:
        None. Saves the plot to plots/data/{output_name}.png
    """
    from data_process import iter_data_items
    from pathlib import Path
    
    # Verify plot folder exists
    verify_plot_folder()
    
    # Collect data and group by meter and key
    grouped = {}
    for item in iter_data_items(Path(data_dir)):
        key = (item.meter, item.key)
        if key not in grouped:
            grouped[key] = 0
        grouped[key] += 1
    
    if not grouped:
        print(f"No data found in {data_dir}")
        return
    
    # Extract unique meters and keys, sorted for consistent ordering
    all_meters = sorted(set(meter for meter, _ in grouped.keys()))
    all_keys = sorted(set(key for _, key in grouped.keys()))
    
    # Create a 2D grid for the heatmap
    grid = np.zeros((len(all_keys), len(all_meters)))
    
    meter_to_idx = {meter: i for i, meter in enumerate(all_meters)}
    key_to_idx = {key: i for i, key in enumerate(all_keys)}
    
    for (meter, key), count in grouped.items():
        grid[key_to_idx[key], meter_to_idx[meter]] = count
    
    # Create the plot
    plt.clf()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use logarithmic scale for better visibility of smaller counts
    grid_log = np.log10(grid + 1)  # Add 1 to avoid log(0)
    
    im = ax.imshow(grid_log, cmap='YlOrRd', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(all_meters)))
    ax.set_yticks(range(len(all_keys)))
    ax.set_xticklabels(all_meters, rotation=45, ha='right', fontsize=20)
    ax.set_yticklabels(all_keys, fontsize=20)
    
    # Add colorbar with appropriate label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Log10(Count + 1)', rotation=270, labelpad=15)
    
    # Add text annotations for counts
    for i, key in enumerate(all_keys):
        for j, meter in enumerate(all_meters):
            count = int(grid[i, j])
            if count > 0:
                text = ax.text(j, i, str(count), ha="center", va="center",
                             color="black" if grid_log[i, j] < 2 else "white",
                             fontsize=20)
    
    # Labels and title
    #ax.set_xlabel('Key', fontsize=12)
    #ax.set_ylabel('Meter', fontsize=12)
    #ax.set_title('Distribution of Key and Meter Pairs in Music Data', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"plots/data/{output_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Grid plot saved to plots/data/{output_name}.png")
    print(f"Total groups: {len(grouped)}")
    print(f"Unique meters: {len(all_meters)}")
    print(f"Unique keys: {len(all_keys)}")

      