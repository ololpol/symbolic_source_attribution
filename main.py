
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

train_data_path = "data/ONeillsJigs_parsed_wot"
output_data_path = "data/folkrnn_out.abc"
embedding = "clamp" # options: "clamp", "clap", "muq", "folkrnn", "random"
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
        data = f.read()


    tokens_set = set(data.split())
    start_symbol, end_symbol = '<s>', '</s>'
    tokens_set.update({start_symbol, end_symbol})

    idx2token = list(tokens_set)
    vocab_size = len(idx2token)
    print('vocabulary size:', vocab_size)
    token2idx = dict(zip(idx2token, range(vocab_size)))
    tunes = data.split('\n\n')

    print(tunes[0])
    print()
    print(idx2token)
    print()
    print(token2idx)

    return tunes, idx2token, token2idx

def format_abc(tune):
    #print(tune)
    pass#TODO either implement this or do it manually

def ABC2wav(tune):
    formatted_abc = format_abc(tune)
    #wav = pysynth.make_wav(formatted_abc) #TODO either implement this or do it manually
    return "data/wav/sessiontune1170.wav"

def load_wav(label = "ONeillsJigs"):
    #TODO check all is good
    wav_fname = f"data/wav/{label}/"
    return wav_fname

def embed(tunes, wav_folder, embedding, use_cache = True, cache_label = ""):
    embedding = embedding.lower()

    # check if the embedded data exists in cache, if so load it instead of recomputing
    if use_cache:
        cache_folder = os.listdir("cache")
        print("Found files in cache:", cache_folder)
        if cache_label != "":
            cache_label = "_"+cache_label
        else: 
            print("No cache label provided")
        cache_file = "embeddings_"+embedding+cache_label+".pkl"

    if use_cache and cache_file in cache_folder:
        with open("cache/"+cache_file, "rb") as f:
            res = pickle.load(f)
        print(embedding+cache_label, "embeddings loaded from cache")
        return res
    elif embedding == "clamp":
        res = clamp(tunes)
    elif embedding == "clap":
        res = clap(wav_folder)
    elif embedding == "muq":
        res = muq(wav_folder)
    elif embedding == "folkrnn":
        res = folkrnn_embed(tunes)
    elif embedding == "random":
        res = [[2*random.random() - 1 for _ in range(512)] for _ in tunes]
    else:
        raise ValueError("Unsupported embedding method: {}".format(embedding))

    # Store res in cache
    with open("cache/" + cache_file, "wb") as f:
        pickle.dump(res, f)
        print("embeddings stored in cache file:", cache_file)

    return res
    
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
    print(f"Loaded {n_files} wav files for MuQ-MuLan embedding")

    # Initialize MuQ-MuLan model
    device = 'cpu'
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
    #TODO actually use this
    with open("muq_embeddings.txt", "w") as f:
        f.write(str(res))
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
    
    res = {}
    for m in methods:
        if m == "euclidean":
            dists = []
            for e2 in data:
                dist = sum([(a-b)**2 for a,b in zip(e1,e2)])**0.5
                dists.append(dist)
        elif m == "cosine":
            # Cosine similarity
            dists = []
            for e2 in data:
                dot_product = sum([a*b for a,b in zip(e1,e2)])
                norm_e1 = sum([a**2 for a in e1])**0.5
                norm_e2 = sum([b**2 for b in e2])**0.5
                dist = 1 - dot_product / (norm_e1 * norm_e2)
                dists.append(dist)
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



if __name__ == "__main__":
    
    # Load abc and wav formats of the data
    tunes, idx2token, token2idx = load_abc(train_data_path)
    outs, _, _ = load_abc(output_data_path)
    wav_fname = load_wav("ONeillsJigs")
    wav_out = load_wav("folkrnn_out")


    # Embed the data using the specified embedding method
    embedded_data = embed(tunes, wav_fname, embedding, cache_label = "ONeill")
    embedded_out = embed(outs, wav_out, embedding, cache_label = "output")
    print("embedded data shape: ", ((len(embedded_data), len(embedded_data[0]))))
    full_data = embedded_data + embedded_out

    plotting.plot_embeddings(full_data, embedding+"full")

    plotting.plot_embeddings(embedded_data, embedding)
    plotting.plot_embeddings_pca(embedded_data, embedding)

    plotting.plot_pca_pairs(embedded_data, embedding)
    plotting.plot_pca_variance(embedded_data, embedding)



    for m in methods:
        plotting.plot_distance_average(embedded_data, embedding, method=m)



    e_out = random.choice(embedded_out) #TODO this should be the embedding of the output tune, not a random one
    out_dists = compute_dist(e_out, embedded_data, methods = methods)
    



    for m in out_dists.keys():
        plotting.plot_distance_distribution(out_dists[m], embedding+"_"+m)
    
    
    

    