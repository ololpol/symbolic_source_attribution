train_data_path = "folkrnn/data/ONeillsJigs_parsed_wot"
embedding = "random" # options: "clamp", "clap", "muq", "folkrnn", "random"

import math
import os
import pysynth
import random
import librosa
import torch
from muq import MuQ, MuQMuLan
import laion_clap
import matplotlib.pyplot as plt


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

def load_wav():
    #TODO check all is good
    wav_fname = "data/wav/"
    print("wav folder detected: ", wav_fname)

    return wav_fname

def embed(tunes, wav_folder, embedding):
    embedding = embedding.lower()
    if embedding == "clamp":
        return clamp(tunes)
    if embedding == "clap":
        return clap(wav_folder)
    if embedding == "muq":
        return muq(wav_folder)
    if embedding == "folkrnn":
        return folkrnn_embed(tunes)
    if embedding == "random":
        return [[2*random.random() - 1 for _ in range(512)] for _ in tunes]
    
def clamp(tunes):
    res = []
    for t in tunes:
        res.append([2*random.random() - 1 for _ in range(100)])
        pass#TODO
    return res

def clap(tune_fname):
    res = []

    fnames = os.listdir(tune_fname)
    tune_fnames = [os.path.join(tune_fname, f) for f in fnames if f.endswith(".wav")]
    print("Loaded wav files for CLAP embedding: ", tune_fnames)

    # Initialize CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.

    n_files = len(tune_fnames)
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
        if i % 10 == 0:
            print("Extracted CLAP embeddings for {} / {} wav files".format(i, n_files))
        i += 1
    return res

def muq(tune_fname):
    res = []

    fnames = os.listdir(tune_fname)
    tune_fnames = [os.path.join(tune_fname, f) for f in fnames if f.endswith(".wav")]
    print("Loaded wav files for MuQ-MuLan embedding: ", tune_fnames)

    # Initialize MuQ-MuLan model
    device = 'cpu'
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()

    n_files = len(tune_fnames)
    print("Extracting MuQ-MuLan embeddings for {} wav files...".format(n_files))
    i = 0
    for wav_fname in tune_fnames:
        wav, sr = librosa.load(wav_fname, sr = 24000)

        # Extract music embeddings
        wav, sr = librosa.load(wav_fname, sr = 24000)
        wavs = torch.tensor(wav).unsqueeze(0).to(device) 
        with torch.no_grad():
            audio_embeds = mulan(wavs = wavs) 

        # Extract text embeddings (texts can be in English or Chinese)
        texts = ["classical genres, hopeful mood, piano.", "一首适合海边风景的小提琴曲，节奏欢快"]
        with torch.no_grad():
            text_embeds = mulan(texts = texts)

        # Calculate dot product similarity
        sim = mulan.calc_similarity(audio_embeds, text_embeds)
        print(sim)
        # Convert audio_embeds to list and append to res
        res.append(audio_embeds[0].cpu().numpy().tolist())

        if i % 10 == 0:
            print("Extracted CLAP embeddings for {} / {} wav files".format(i, n_files))
        i += 1
    return res

def folkrnn_embed(tunes):
    res = []
    for t in tunes:
        res.append([2*random.random() - 1 for _ in range(100)])
        pass#TODO
    return res

def compute_dist(e1, e2, methods = [], method = None):
    if methods != [] and method != None:
        raise ValueError("Either methods or method should be provided, not both")
    if methods == []:
        if method == None:
            raise ValueError("Either methods or method should be provided")
        methods = [method]
    
    res = {}
    for m in methods:
        if m == "euclidean":
            dist = sum([(a-b)**2 for a,b in zip(e1,e2)])**0.5
        elif m == "cosine":
            # Cosine similarity
            dot_product = sum([a*b for a,b in zip(e1,e2)])
            norm_e1 = sum([a**2 for a in e1])**0.5
            norm_e2 = sum([b**2 for b in e2])**0.5
            dist = 1 - dot_product / (norm_e1 * norm_e2)
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
        elif m == "Tversky":
            # Tversky index
            pass # For sets
        else:
            raise ValueError("Unsupported distance method: {}".format(m))
        res[m] = dist
    return res



if __name__ == "__main__":
    
    # Load abc and wav formats of the data
    tunes, idx2token, token2idx = load_abc(train_data_path)
    wav_fname = load_wav()


    # Embed the data using the specified embedding method
    embedded_data = embed(tunes, wav_fname, embedding)
    print("embedded data shape: ", ((len(embedded_data), len(embedded_data[0]))))

    
    #TODO plot embeddings


    #TODO pick/generate output?
    output = 0 
    e_out = random.choice(embedded_data) #TODO this should be the embedding of the output tune, not a random one
    #e_out = embed([output], None, embedding)[0]



    m = "cosine"
    out_dists = []
    for e in embedded_data:
        dist = compute_dist(e, e_out, method = m)
        out_dists.append(dist[m])

        

    # Plot dists_sorted as a bar chart, x axis is distance threshould, y axis is nr of distances
    dists_sorted = sorted(out_dists)
    
    t_step = 0.05
    t_steps = math.ceil(1/t_step)
    thresholds = [t_step*i for i in range(t_steps+1)]
    thresholds[0] = -0.001 # to include the 0 distance

    dist_counts = [sum([1 for d in out_dists if d > t and d <= t+t_step]) for t in thresholds]

    plt.bar(thresholds, dist_counts, width=t_step)
    plt.xlabel("Distance threshold")
    plt.ylabel("Number of distances")
    plt.title("Distribution of distances to output embedding")
    plt.savefig("plots/distance_distribution.png")
    
    

    