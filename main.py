
import math
import os
import pysynth
import random
import librosa
import torch
from muq import MuQ, MuQMuLan
import laion_clap
import matplotlib.pyplot as plt
import plotting
from clamp_utils import *
from transformers import AutoTokenizer
import pickle
import numpy as np
import scipy

target_data_labels = ["ONeill", "output", "Folkwiki","Folkwiki_ONeill_10","Folkwiki_ONeill_20","Folkwiki_ONeill_30","Folkwiki_ONeill_40","Folkwiki_ONeill_50","Folkwiki_ONeill_60","Folkwiki_ONeill_70","Folkwiki_ONeill_80","Folkwiki_ONeill_90","Folkwiki_ONeill_100"] #TODO add more labels here as needed, these should correspond to the folder names in data/wav/
embeddings = ["clamp"] # options: "clamp", "clap", "muq", "folkrnn", "random"
methods = ["cosine"] # options: "euclidean", "cosine", "cl", "matching", "hamming", "jaccard", "orchini", "sorencen-dice", "tanimoto", "tucker", "tversky"
CLAMP_MODEL_NAME = "sander-wood/clamp-small-512"


if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def load_abc(data_path):
    with open(data_path, 'r') as f:
        data = f.read().strip()


    tokens_set = set(data.split())
    start_symbol, end_symbol = '<s>', '</s>'
    tokens_set.update({start_symbol, end_symbol})

    idx2token = list(tokens_set)
    vocab_size = len(idx2token)
    print('vocabulary size:', vocab_size)
    token2idx = dict(zip(idx2token, range(vocab_size)))
    tunes = data.split('\n\n')


    return tunes, idx2token, token2idx

def format_abc(tune):
    #print(tune)
    pass#TODO either implement this or do it manually

def ABC2wav(tune):
    formatted_abc = format_abc(tune)
    #wav = pysynth.make_wav(formatted_abc) #TODO either implement this or do it manually
    return "data/wav/sessiontune1170.wav"

def load_wav(label = "ONeill"):
    #TODO check all is good
    wav_fname = f"data/wav/{label}/"
    return wav_fname

def embed(data, embedding, use_cache = True, cache_label = ""):
    embedding = embedding.lower()

    # check if the embedded data exists in cache, if so load it instead of recomputing
    if use_cache:
        cache_folder = os.listdir("cache")
        #print("Found files in cache:", cache_folder)
        if cache_label != "":
            cache_label = "_"+cache_label
        else: 
            print("No cache label provided")
        cache_file = "embeddings_"+embedding+cache_label+".pkl"

    if use_cache and cache_file in cache_folder:
        with open("cache/"+cache_file, "rb") as f:
            res = pickle.load(f)
        print(embedding+cache_label, "embeddings loaded from cache")
        return np.array(res)
    elif embedding == "clamp":
        res = clamp(data["abc"])
    elif embedding == "clap":
        res = clap(data["wav"])
    elif embedding == "muq":
        res = muq(data["wav"])
    elif embedding == "folkrnn":
        res = folkrnn_embed(data["abc"])
    elif embedding == "random":
        res = [[2*random.random() - 1 for _ in range(512)] for _ in data["abc"]]
    else:
        raise ValueError("Unsupported embedding method: {}".format(embedding))

    # Store res in cache
    with open("cache/" + cache_file, "wb") as f:
        pickle.dump(res, f)
        print("embeddings stored in cache file:", cache_file)

    return np.array(res)
    
def clamp(tunes):
    res = []


    # Initialize CLAMP model
    # load CLaMP model
    clamp_model = CLaMP.from_pretrained(CLAMP_MODEL_NAME)

    music_length = 1024 #model.config.max_length
    clamp_model = clamp_model.to(device)
    clamp_model.eval()

    # initialize patchilizer, and softmax

    patchilizer = MusicPatchilizer()
    softmax = torch.nn.Softmax(dim=1)


    for t in tunes:
        query = load_music(data=t)
        query_ids = encoding_data([query], patchilizer, music_length)
        query_feature = get_features(query_ids, clamp_model, device)
        #print(query_feature)
        res.append(query_feature.cpu().numpy().tolist()[0])


    return res

def clap(tune_fname):
    res = []

    fnames = os.listdir(tune_fname)
    tune_fnames = [os.path.join(tune_fname, f) for f in fnames if f.endswith(".wav")]
    n_files = len(tune_fnames)

    print(f"Loaded {n_files} wav files for CLAP embedding")

    # Initialize CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.

    print("Extracting CLAP embeddings for {} wav files...".format(n_files))
    i = 0
    for wav_fname in tune_fnames:
        # Get audio embeddings from audio data
        audio_data, _ = librosa.load(wav_fname, sr=48000) # sample rate should be 48000
        audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
        audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
        #print("Audio embed first 20:", audio_embed[:,-20:])
        #print("Audio embed shape:", audio_embed.shape)
        res.append(audio_embed[0])
        i += 1
        if i % 10 == 0:
            print("Extracted CLAP embeddings for {} / {} wav files".format(i, n_files))
    return res

def muq(tune_fname):
    res = []

    fnames = os.listdir(tune_fname)
    tune_fnames = [os.path.join(tune_fname, f) for f in fnames if f.endswith(".wav")]
    n_files = len(tune_fnames)

    print(f"Loaded {n_files} wav files for MuQ-MuLan embedding")

    # Initialize MuQ-MuLan model
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    n_files = len(tune_fnames)
    print("Extracting MuQ-MuLan embeddings for {} wav files...".format(n_files))
    i = 0
    for wav_fname in tune_fnames:

        # Extract music embeddings
        wav, sr = librosa.load(wav_fname, sr = 24000)
        wavs = torch.tensor(wav).unsqueeze(0).to(device) 
        with torch.no_grad():
            audio_embeds = mulan(wavs = wavs) 

        # Convert audio_embeds to list and append to res
        res.append(audio_embeds[0].cpu().numpy().tolist())

        i += 1
        if i % 10 == 0:
            print("Extracted MuQ-MuLan embeddings for {} / {} wav files".format(i, n_files))
    return res

def folkrnn_embed(tunes):
    res = []
    for t in tunes:
        res.append([2*random.random() - 1 for _ in range(100)])
        pass#TODO
    return res

def compute_dist(e1, data, methods = [], method = None):
    if methods != [] and method != None:
        raise ValueError("Either methods or method should be provided, not both")
    if methods == []:
        if method == None:
            raise ValueError("Either methods or method should be provided")
        methods = [method]
    
    res = {} #TODO idk if this helps
    for m in methods:
        if m == "euclidean":
            # TODO do this with numpy for efficiency
            dists = []
            for e2 in data:
                dist = sum([(a-b)**2 for a,b in zip(e1,e2)])**0.5
                dists.append(np.array(dist))
        elif m == "cosine":
            # Cosine distances
            # TODO do this with numpy for efficiency
            dists = np.zeros(len(data))
            for i, e2 in enumerate(data):
                
                dist = scipy.spatial.distance.cosine(e1, e2)
                dists[i] = dist
                
                
                #dot_product = sum([a*b for a,b in zip(e1,e2)])
                #norm_e1 = sum([a**2 for a in e1])**0.5
                #norm_e2 = sum([b**2 for b in e2])**0.5
                #dist = 1 - dot_product / (norm_e1 * norm_e2)
                #dists.append(np.array(dist))
        elif m == "cl":
            #Contrastive learning encoding distance
            pass
        elif m == "matching":
            # Simple Matching Coefficient
            pass # binary vectors
        elif m == "hamming":
            # Hamming distance
            pass # binary vectors?
        elif m == "jaccard":
            # Jaccard index
            pass # For sets?
        elif m == "orchini":
            # Orchini similarity
            pass # i guess this is just cosine similarity?
        elif m == "sorencen-dice":
            # F1 score?
            pass
        elif m == "tanimoto":
            # Tanimoto distance
            pass #binary sets?
        elif m == "tucker":
            # Tucker coefficient of congruence
            pass # i guess this is just cosine similarity?
        elif m == "tversky":
            # Tversky index
            pass # For sets
        else:
            raise ValueError("Unsupported distance method: {}".format(m))
        res[m] = dists
    return res


def compute_attribution(data, output_labels, data_labels = None, attribution_method = None, dist_method = "cosine", embedding = "clamp", temperature = 1):
    # Compute attribution scores the output embedding with respect to the data embeddings using the specified method
    
    #TODO more ways to do this
    
    id_map = {}
    label_map = {}
    ids = []
    ns = []
    n_artists = len(data_labels)
    embeddings = None
    i = 0
    for label in data_labels:
        if embeddings is None:
            embeddings = data[label]["embed"][embedding]
        else:
            embeddings = np.vstack([embeddings, data[label]["embed"][embedding]])
        id_map[i] = label
        label_map[label] = i
        ids += [i for _ in data[label]["embed"][embedding]]
        ns.append(len(data[label]["embed"][embedding]))
        i += 1

    outputs = None
    for label in output_labels:
        if outputs is None:
            outputs = data[label]["embed"][embedding]
        else:
            outputs = np.vstack([outputs, data[label]["embed"][embedding]])
    

    res = np.zeros((len(outputs), n_artists))
    for j, output in enumerate(outputs):
        dists = compute_dist(output, embeddings, method=dist_method)[dist_method]

        sims = np.exp(-np.array(dists)/temperature)

        attribution = np.zeros(n_artists)
        for i in range(len(sims)):
            attribution[ids[i]] += sims[i]
        
        for i in range(len(attribution)):
            attribution[i] /= ns[i] #normalize per artist

        res[j] = attribution
    
    res = scipy.special.softmax(res, axis=1) #normalize across artists

    return res, id_map, label_map




if __name__ == "__main__":
    
    # Load abc and wav formats of the data
    data = {}

    for label in target_data_labels:
        data[label] = {}
        if os.path.exists(f"data/{label}.abc"):
            data[label]["abc"], _, _ = load_abc(f"data/{label}.abc")
        elif os.path.exists(f"data/{label}.txt"):
            data[label]["abc"], _, _ = load_abc(f"data/{label}.txt")
        else:
            raise ValueError(f"No abc or txt file found for label {label} in data folder")
        data[label]["wav"] = load_wav(label)
        data[label]["embed"] = {}
        for embedding in embeddings:
            embedded_data = embed(data[label], embedding, cache_label = label)
            data[label]["embed"][embedding] = embedded_data
            print(data[label]["embed"][embedding].shape)
            embed_len = len(embedded_data[0])
            data[label]["avg_embed"] = np.average(embedded_data, axis=0)
            #data[label]["avg_embed"] = [sum([e[i] for e in embedded_data])/len(embedded_data) for i in range(embed_len)]



    for embedding in embeddings:
        # Embed the data using the specified embedding method

        plotting.plot_embeddings(data, "ONeill", embedding)
        plotting.plot_embeddings_pca(data, "ONeill", embedding)

        plotting.plot_pca_pairs(data, "ONeill", embedding)
        plotting.plot_pca_variance(data, "ONeill", embedding, components=100)



        for m in methods:
            break
            plotting.plot_distance_average(data, "ONeill", "ONeill", embedding, method=m)
            plotting.plot_distance_average(data, ["ONeill", "output"], ["ONeill", "output"], embedding, method=m)
            plotting.plot_distance_average(data, "ONeill", "output", embedding, method=m)
            plotting.plot_distance_average(data, "output", "ONeill", embedding, method=m)



        e_out = random.choice(data["output"]["embed"][embedding])

        out_dists = compute_dist(e_out, data["ONeill"]["embed"][embedding], methods = methods)
        out_dists_full = compute_dist(e_out, np.vstack([data["ONeill"]["embed"][embedding], data["output"]["embed"][embedding]]), methods = methods)



        for m in out_dists.keys():
            plotting.plot_origin_distance(data, "ONeill", embedding, method="euclidean", t_step=0.01)
            plotting.plot_centroid_distance(data, "ONeill", embedding, method="euclidean", t_step=0.01)
            plotting.plot_distance_distribution(out_dists[m], embedding+"_"+m, t_step=0.01)
            plotting.plot_distance_distribution(out_dists_full[m], embedding+"_full_"+m, t_step=0.01)

        

        #TODO look at temperature


        for label in target_data_labels:
                
            target_labels = target_data_labels.copy()
            target_labels.remove(label)
            attribution, id_map, label_map = compute_attribution(data, output_labels=[label], data_labels = target_labels, attribution_method = None, dist_method = "cosine", embedding = embedding, temperature = 0.1)

            
            plotting.plot_attribution(random.choice(attribution), id_map, label + "_" + embedding)
            plotting.plot_attribution_distribution(data, [label], target_labels, temperature = 0.1)

        
        
    

    