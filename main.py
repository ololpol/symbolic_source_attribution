"""
This is the main file that performs all the experiments.
If the neccecarry previous files aren't ran, this file runs them.
"""
import math
import os
import pysynth
import random
import librosa
import torch
from muq import MuQ, MuQMuLan
import matplotlib.pyplot as plt
import plotting
from clamp_utils import *
from transformers import AutoTokenizer
import pickle
import numpy as np
import scipy
import subprocess
import glob
import shutil

secondary_labels = []  #Labels that should not be used to generate main plots
#target_data_labels = ["ONeill", "output", "Folkwiki"]
embeddings = ["clamp"] # options: "clamp", "clap", "muq", "folkrnn", "random"
methods = ["cosine"] # options: "euclidean", "cosine", "cl", "matching", "hamming", "jaccard", "orchini", "sorencen-dice", "tanimoto", "tucker", "tversky"
CLAMP_MODEL_NAME = "sander-wood/clamp-small-512"

random.seed(61)
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def verify_folder_structure():
    """
    Verifies that all neccessary folders exist, and creates them if they don't. This includes the "data", "data/midi", "data/wav", and "cache" folders. Additionally, it calls the plotting.verify_plot_folder() function to ensure that the necessary plotting folders are present.
    Args:
        None
    Returns:
        None
    """
    # Check if the necessary folders exist, and create them if they don't
    folders = ["data", "data/midi", "data/wav", "cache"]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    plotting.verify_plot_folder()

    #TODO check if a set of sub folders exist in the data folder, and create them if they don't


def load_abc(data_path):
    """
    Loads a file with tokenized (space-separated) ABC data and returns the tunes, and token maps. The function reads the file, splits the data into tokens, and creates a set of unique tokens.
    Args:
        data_path (str): The path to the ABC data file.
    Returns:
        tunes (list): A list of tunes extracted from the ABC data.
        idx2token (list): A list mapping indices to tokens.
        token2idx (dict): A dictionary mapping tokens to indices.
    """
    with open(data_path, 'r') as f:
        data = f.read().strip()


    tokens_set = set(data.split())
    start_symbol, end_symbol = '<s>', '</s>'
    tokens_set.update({start_symbol, end_symbol})

    idx2token = list(tokens_set)
    vocab_size = len(idx2token)
    token2idx = dict(zip(idx2token, range(vocab_size)))
    tunes = data.split('\n\n')


    return tunes, idx2token, token2idx

def load_folder(folder_path):
    """
    Finds all .abc or .txt files in the folder and returns a list of labels corresponding to the files found.
    Args:
        folder_path (str): The path to the folder containing the .abc or .txt files.
    Returns:
        labels (list): A list of labels corresponding to the .abc or .txt files found in the folder.
    """
    labels = []
    for filename in os.listdir(folder_path):
        folder_name = os.path.split(folder_path)[1]
        if filename.endswith(".abc") or filename.endswith(".txt"):
            label = os.path.splitext(filename)[0]
            labels.append(folder_name + "/" + label)
    print("Found labels in folder '{}': {}".format(folder_path, labels))
    return labels


def format_abc(filename):
    """
    Formats the tunes in a file in a way that can be processed by abc2midi. 
    Args:
        filename (str): The path to the file containing the tunes to be formatted.
    Returns:
        None
    """
    subprocess.run(["python", "abc_processing.py", "--data_path", filename])


def process_abc(labels, model_name="clap"):
    """
    Processes a label by finding its ABC/txt file, converting to MIDI,
    then to WAV, and applying a specified embedding function to each WAV output.
    
    Args:
        label: The label name to process
        
    Returns:
        A list of results from applying apply() to each WAV file
    """
    # Initialize model
    if model_name == "clap":
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt() # download the default pretrained checkpoint.



    elif model_name == "muq":
        model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
        model = model.to(device).eval()

    for label in labels:
        print("Starting ABC to wav processing for label:", label)

        # 1. Look for <label>.txt or <label>.abc in the data folder
        txt_path = os.path.join("data", f"{label}.txt")
        abc_path = os.path.join("data", f"{label}.abc")
        txt_path_labelled = os.path.join("data", f"{label}_labelled.txt")
        abc_path_labelled = os.path.join("data", f"{label}_labelled.abc")

        
        if os.path.exists(abc_path_labelled):
            filename = abc_path_labelled
        elif os.path.exists(txt_path_labelled):
            filename = txt_path_labelled
        elif os.path.exists(abc_path):
            print("formatting from ", abc_path)
            format_abc(abc_path)
            filename = abc_path_labelled
        elif os.path.exists(txt_path):
            print("formatting from ", txt_path)
            format_abc(txt_path)
            filename = txt_path_labelled
        else:
            raise FileNotFoundError(
                f"No file found for label '{label}'. "
                f"Expected 'data/{label}.abc' or 'data/{label}.txt'."
                f"Or labelled data {abc_path_labelled} or {txt_path_labelled}"
            )
        print("starting ABC2midi")
        
        # 2. Run abc2midi on the found file
        result = subprocess.run(
            ["abcmidi/abc2midi", filename],
            capture_output=True,
            text=True
        )
        #if result.returncode != 0:
        #    raise RuntimeError(
        #        f"abc2midi failed for '{filename}':\n{result.stderr}"
        #    )
        
        # 3. Check for/create the output MIDI folder
        midi_dir = os.path.join("data", "midi", label)
        os.makedirs(midi_dir, exist_ok=True)
        
        # 4. Move all generated .mid files into the folder
        mid_files = glob.glob(os.path.join("data", "*.mid"))
        if not mid_files:
            raise RuntimeError("abc2midi ran successfully but produced no .mid files.")
        
        moved_mid_files = []
        for mid_file in mid_files:
            dest = os.path.join(midi_dir, os.path.basename(mid_file))
            shutil.move(mid_file, dest)
            moved_mid_files.append(dest)


        # 5 & 6. Convert each .mid to WAV with timidity, then apply model to the output
        res = []

        print("Starting processing of", len(moved_mid_files), "files")
        i = 0
        for mid_file in moved_mid_files:
            wav_file = "timidity_temp.wav"
            
            timidity_result = subprocess.run(
                ["timidity", mid_file, "-Ow", "-o", wav_file],
                capture_output=True,
                text=True
            )
            if timidity_result.returncode != 0:
                raise RuntimeError(
                    f"timidity failed for '{mid_file}':\n{timidity_result.stderr}"
                )
            
            # Read the WAV data and apply the processing function
            with open(wav_file, "rb") as f:
                wav_data = f.read()

            if model_name=="clap":
                # Get audio embeddings from audio data
                audio_data, _ = librosa.load(wav_file, sr=48000) # sample rate should be 48000
                audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
                audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
                #print("Audio embed first 20:", audio_embed[:,-20:])
                #print("Audio embed shape:", audio_embed.shape)
                res.append(audio_embed[0])
            elif model_name == "muq":
                # Extract music embeddings
                wav, sr = librosa.load(wav_file, sr = 24000)
                wavs = torch.tensor(wav).unsqueeze(0).to(device) 
                with torch.no_grad():
                    audio_embeds = model(wavs = wavs) 

                # Convert audio_embeds to list and append to res
                res.append(audio_embeds[0].cpu().numpy().tolist())

            i += 1
            if i % 10 == 0:
                print(i, "files processed")
        
            # Store res in cache
        if model_name == "clap" or model_name == "muq":
            cache_label = model_name + "_" + label
            cache_file = "embeddings_"+cache_label+".pkl"
            with open("cache/" + cache_file, "wb") as f:
                pickle.dump(res, f)
                print("embeddings stored in cache file:", cache_file)



def assure_wav(label = "ONeill"):
    #TODO use this?
    wav_fname = f"data/wav/{label}/"

    if not os.path.exists(wav_fname):
        raise FileNotFoundError(f"No wav folder found for label '{label}'. Expected at '{wav_fname}'")

    # Check if wav folder is empty
    folder_files = os.listdir(wav_fname)
    folder_files = [f for f in folder_files if f.endswith(".wav")]
    if len(folder_files) == 0:
        raise FileNotFoundError(f"Wav folder for label '{label}' is empty. Expected wav files in '{wav_fname}'")


    return wav_fname



def extract_wav(label, id):
    """
    Creates a wav file from a midi file for a given label and id. Uses timidity to convert the midi file to wav format. The resulting wav file is saved in the appropriate folder.
    Args:
        label (str): The label in which the midi file is located.
        id (int): The id of the tune within the label.
    Returns:
        wav_fname (str): The path to the created wav file.
    """
    midi_fname = f"data/midi/{label}/{label}_labelled{id}.mid"
    if not os.path.exists(midi_fname):
        raise FileNotFoundError(f"No wav file found for label '{label}' and id '{id}'. Expected at '{midi_fname}'")
    if not os.path.exists(f"data/wav/{label}/"):
        os.makedirs(f"data/wav/{label}/", exist_ok=True)
    wav_fname = f"data/wav/{label}/{label}{id}.wav"

    timidity_result = subprocess.run(
                ["timidity", midi_fname, "-Ow", "-o", wav_fname],
                capture_output=True,
                text=True
            )
    if timidity_result.returncode != 0:
        raise RuntimeError(
            f"timidity failed for '{midi_fname}':\n{timidity_result.stderr}"
        )

    #print(f"wav file extracted for label '{label}' and id '{id}'")

    return wav_fname

def embed(data, embedding, use_cache = True, cache_label = ""):
    """
    Embeds the data using the specified embedding method. If use_cache is True, it will check if the embeddings have already been computed and stored in a cache file. If so, it will load the embeddings from the cache instead of recomputing them.
    Args:
        data (dict): A dictionary containing the data to be embedded. It should have keys "abc" for ABC data and "wav" for WAV data.
        embedding (str): The embedding method to use. Options are "clamp", "clap", "muq", or "random".
        use_cache (bool): Whether to use cached embeddings if available. Default is True.
        cache_label (str): A label to specify the cache file name. 
    Returns:
        np.array: An array of embedded data.
    """
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
        #print(embedding+cache_label, "embeddings loaded from cache")
        return np.array(res)
    elif embedding == "clamp":
        print("Embedding", cache_label, "with CLAMP model")
        res = clamp(data["abc"])
    elif embedding == "clap":
        print("Embedding", cache_label, "with CLAP model")
        res = clap(data["wav"])
    elif embedding == "muq":
        print("Embedding", cache_label, "with MuQ-MuLan model")
        res = muq(data["wav"])
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
    """
    Embeds the tunes using the CLaMP model. 
    Args:
        tunes (list): A list of tunes in ABC format to be embedded.
    Returns:
        res (list): A list of embedded tunes.
    """
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
        #remove spaces from t, TODO this can be done earier since all models want this
        t = t.replace(" ", "")
        print("prepring tune for CLAMP: ", t)

        query = load_music(data=t)
        query_ids = encoding_data([query], patchilizer, music_length)
        query_feature = get_features(query_ids, clamp_model, device)
        #print(query_feature)
        res.append(query_feature.cpu().numpy().tolist()[0])


    return res

def clap(tune_fname):
    """
    Embeds the tunes using the CLAP model. 
    Args:
        tune_fname (str): The path to the directory containing the tune files.
    Returns:
        res (list): A list of embedded tunes.
    """
    import laion_clap

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
        res.append(audio_embed[0])
        i += 1
        if i % 10 == 0:
            print("Extracted CLAP embeddings for {} / {} wav files".format(i, n_files))
    return res

def muq(tune_fname):
    """
    Embeds the tunes using the MuQ-MuLan model. 
    Args:
        tune_fname (str): The path to the directory containing the tune files.
    Returns:
        res (list): A list of embedded tunes.
    """
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



def embed_all(data, embeddings):
    """
    Embeds all data in the provided dictionary using the specified embedding methods. The embedding is done for all keys in the data directory.
    Args:
        data (dict): A dictionary containing the data to be embedded. It should have keys corresponding to the labels in the data directory.
        embeddings (list): A list of embedding methods to use. Options are "clamp", "clap", "muq", or "random".
    Returns:
        data (dict): The input data dictionary with the embedded data added under each label and embedding method.
    """
    for label in data.keys():
        for embedding in embeddings:
            embedded_data = embed(data[label], embedding, cache_label = label)
            data[label][embedding] = embedded_data

            #print(label, data[label][embedding].shape)
            #embed_len = len(embedded_data[0])
            data[label]["avg_embed"] = np.average(embedded_data, axis=0)

    return data

def compute_dist(source, data, method):
    """
    Computes distances between two sets of embeddings.
    Args: 
        source (list): A list of source embeddings.
        data (list): A list of target embeddings.
        method (str): The distance computation method to use. Options are "euclidean" or "cosine"
    Returns:
        dists (np.array): A 2D array of distances between each source embedding and each target embedding.
    """
    if isinstance(source[0], float) or isinstance(source[0], int):
        source = [source]

    
    if method == "euclidean":
        # TODO do this with numpy for efficiency
        dists = []
        for e1 in source:
            dists_d = []
            for e2 in data:
                dist = sum([(a-b)**2 for a,b in zip(e1,e2)])**0.5
                dists_d.append(dist)
            dists.append(dists_d)
        dists = np.array(dists)
    elif method == "cosine":
        # Cosine distances
        # TODO do this with numpy for efficiency
        dists = np.zeros((len(source), len(data)))
        for i, e1 in enumerate(source):
            for j, e2 in enumerate(data):
                dist = scipy.spatial.distance.cosine(e1, e2)
                dists[i, j] = dist
            
            
            #dot_product = sum([a*b for a,b in zip(e1,e2)])
            #norm_e1 = sum([a**2 for a in e1])**0.5
            #norm_e2 = sum([b**2 for b in e2])**0.5
            #dist = 1 - dot_product / (norm_e1 * norm_e2)
            #dists.append(np.array(dist))
    elif method == "cl":
        #Contrastive learning encoding distance
        pass
    elif method == "matching":
        # Simple Matching Coefficient
        pass # binary vectors
    elif method == "hamming":
        # Hamming distance
        pass # binary vectors?
    elif method == "jaccard":
        # Jaccard index
        pass # For sets?
    elif method == "orchini":
        # Orchini similarity
        pass # i guess this is just cosine similarity?
    elif method == "sorencen-dice":
        # F1 score?
        pass
    elif method == "tanimoto":
        # Tanimoto distance
        pass #binary sets?
    elif method == "tucker":
        # Tucker coefficient of congruence
        pass # i guess this is just cosine similarity?
    elif method == "tversky":
        # Tversky index
        pass # For sets
    else:
        raise ValueError("Unsupported distance method: {}".format(m))

    return dists


def compute_attribution(data, source_labels, target_labels, attribution_method = None, dist_method = "cosine", embedding = "clamp",
    top_N = None, dist_threshold = None, top_Y = None, attribution_threshold = None, extract_N = None, extract_write = True):
    """
    Computes the attribution from tunes in source_labels to tunes in target_labels using the specified embedding and distance methods. 
    Args:
        data (dict): A dictionary containing the data to be used for attribution. It should have keys corresponding to the labels in the data directory.
        source_labels (list): A list of labels in the data for which to compute attribution from.
        target_labels (list): A list of labels in the data for which to compute attribution to.
        attribution_method (str): The method to use for computing attribution. Currently not implemented.
        dist_method (str): The distance computation method to use. Options are "euclidean" or "cosine". Default is "cosine".
        embedding (str): The embedding method to use. Options are "clamp", "clap", "muq", or "random". Default is "clamp".
        top_N (int): If specified, only the top N closest target tunes to each tune will be considered for attribution computation.
        dist_threshold (float): If specified, only target tunes within this distance threshold will be considered for each source tune attribution computation
        top_Y (int): If specified, only the top Y artists with the highest attribution scores will be returned in the final attribution scores.
        attribution_threshold (float): If specified, only artists with an attribution score above this threshold will be considered in the final attribution scores.
        extract_N (int): If specified, extracts the top N closest items for each output embedding with distance threshold dist_threshold. Default is None.
        extract_write (bool): If True, writes the extracted items to a file. Default is True.
    """
    #TODO more ways to do this
    
    id_map = {}
    label_map = {}
    ids = []
    ns = []


    n_artists = len(target_labels)
    embeddings = None
    i = 0

    if target_labels is None:
        target_labels = source_labels

    # Flatten the data embeddings from different labels/artists and create id and label maps
    for label in target_labels:
        if embeddings is None:
            embeddings = data[label][embedding]
        else:
            embeddings = np.vstack([embeddings, data[label][embedding]])
        id_map[i] = label
        label_map[label] = i
        ids += [i for _ in data[label][embedding]]
        ns.append(len(data[label][embedding]))
        i += 1



    
    #print("id_map: ", id_map)
    #print("label_map: ", label_map)
    #print("ids: ", ids)
    #print("ns: ", ns)

    outputs = None
    for label in source_labels:
        if outputs is None:
            outputs = data[label][embedding]
        else:
            outputs = np.vstack([outputs, data[label][embedding]])
    #print("embeddings shape: ", embeddings.shape)

    #print("outputs shape: ", outputs.shape)
    

    res = np.zeros((len(outputs), n_artists))
    dists = None
    dists = compute_dist(outputs, embeddings, method=dist_method)
    #for j, output in enumerate(outputs):
    #    dists_j = compute_dist(output, embeddings, method=dist_method)
    #    if dists is None:
    #        dists = np.array(dists_j)
    #    else:
    #        dists = np.vstack([dists, dists_j])
    #print("dists shape: ", dists.shape)


    dists_flat = dists.flatten()
    #print("Distance stats: min {}, max {}, mean {}, median {}".format(np.min(dists_flat), np.max(dists_flat), np.mean(dists_flat), np.median(dists_flat)))
    

    if extract_N is not None:
        extracted = []
        if extract_write:
            out_filename = "_".join(source_labels) + ".txt"
            out_filename = "plots/distances/" + out_filename
            out_file = open(out_filename, "w")
            out_file.write("Extracting top {} closest items for each output embedding with distance threshold {}\n\n".format(extract_N, dist_threshold))
            out_file.write("Distance stats: min {}, max {}, mean {}, median {}\n\n".format(np.min(dists_flat), np.max(dists_flat), np.mean(dists_flat), np.median(dists_flat)))
            target_string = "_".join(target_labels)
            source_string = "_".join(source_labels)
            out_file.write("Evaluating distances from {} to {} using embedding '{}' and distance method '{}'\n\n".format(source_string, target_string, embedding, dist_method))
        
    

    for j, output in enumerate(outputs):
        #sims = np.exp(-np.array(dists)/temperature)
        dist_j = dists[j]
        #TODO check this
        #cos theta + 1
        #e ^ (1-d)
        #1 - d

        # top N of distances, per item distance thresold
        if extract_N is not None:
            extract_j = []
            if extract_write:
                out_file.write(f"Item {j+1}\n")
            top_indices = np.argsort(dist_j)[:extract_N]
            for e in top_indices:
                id = ids[e]
                label = id_map[id]
                pos = e - sum(ns[:id]) + 1
                if extract_write:
                    extract_wav(label, pos)
                    out_file.write(f"label '{label}' and id '{pos}'\n")
                    out_file.write("Dist: " + str(dist_j[e]) + "\n")
                extract_j.append((label, pos, dist_j[e]))
            extracted.append(extract_j)

        if dist_threshold is not None:
            dist_j[dist_j > dist_threshold] = 1 # Set distances above the threshold to 1 (max distance)
        if top_N is not None:
            top_indices = np.argsort(dist_j)[:top_N]
            # Set all distances not in the top N to 1 (max distance)
            mask = np.ones_like(dist_j, dtype=bool)
            mask[top_indices] = False
            dist_j[mask] = 1
        


        #TODO other methods to convert this     
        sims_j = 1 - dist_j 
        #print(f"sims_j stats: min {np.min(sims_j)}, max {np.max(sims_j)}, mean {np.mean(sims_j)}, median {np.median(sims_j)}")
        #print("Nr of nonzero sims_j:", np.sum(sims_j > 0))

        #print("sims shape:", sims_j.shape)
        attribution = np.zeros(n_artists)
        for i in range(len(sims_j)):
            #if (sims_j[i] > 0):
            #    print("attribution", ids[i],"increased by", sims_j[i], "element", i)
            attribution[ids[i]] += sims_j[i]
        
        #print("raw attribution for output {}: {}".format(j, attribution))
        for i in range(len(attribution)):
            attribution[i] /= ns[i] #normalize per artist

        #print("pre softmax: ", attribution)

        if np.sum(attribution) > 0:
            attribution = attribution / np.sum(attribution) #normalize across artists
        else:
            print("Warning: attribution sum is 0. Cannot normalize")
        #attribution = scipy.special.softmax(attribution) #normalize across artists

        # top Y artists, per artist threshold attribution
        if attribution_threshold is not None:
            attribution = attribution * (attribution > attribution_threshold) #Set attribution to 0 if it is below the threshold
        if top_Y is not None:
            top_indices = np.argsort(attribution)[-top_Y:]
            for i in range(len(attribution)):
                if i not in top_indices:
                    attribution[i] = 0
        
        if top_Y is not None or attribution_threshold is not None:
            attribution = attribution / np.sum(attribution) #normalize across artists


        #print("post softmax: ", attribution)
        res[j] = attribution
    


    if extract_N is not None and extract_write:
        out_file.close()
    if extract_N is not None:
        return res, id_map, label_map, extracted
    return res, id_map, label_map

def get_data(labels):
    """
    Loads the data for the specified labels.
    Args:
        labels (list): A list of labels for which to load the data. Each label should correspond a file in the "data" folder with either a .abc or .txt extension.
    Returns:
        data (dict): A dictionary where each key is a label and the value is another dictionary containing the loaded ABC data and the path to the corresponding WAV files.
    """
    data = {}

    
    #process_abc(secondary_labels, model_name="muq")
    #process_abc(secondary_labels, model_name="clap")

    for label in labels:
        label_name = label.split("/")[-1]
        data[label_name] = {}
        if os.path.exists(f"data/{label}.abc"):
            data[label_name]["abc"], _, _ = load_abc(f"data/{label}.abc")
        elif os.path.exists(f"data/{label}.txt"):
            data[label_name]["abc"], _, _ = load_abc(f"data/{label}.txt")
        else:
            raise ValueError(f"No abc or txt file found for label {label} in data folder")
        data[label_name]["wav"] = f"data/wav/{label}/"

    return data
    



if __name__ == "__main__":

    verify_folder_structure()
    if not os.path.exists("data/grouped_both") or not os.path.exists("data/grouped_meter") or not os.path.exists("data/grouped_key"):
        #run data_process.py if not ran to create grouped training data
        subprocess.run(["python", "data_process.py"])
    if not os.path.exists("data/folkrnn_both") or not os.path.exists("data/folkrnn_meter") or not os.path.exists("data/folkrnn_key"):
        #run output_data_parse.py if not ran to create grouped output data
        subprocess.run(["python", "output_data_parse.py"])
    if not os.path.exists("data/modified"):
        #run data_modification.py if not ran to create modified data
        subprocess.run(["python", "data_modification.py"])

    data_both = load_folder("data/grouped_both")
    data_meter = load_folder("data/grouped_meter")
    data_key = load_folder("data/grouped_key")

    output_both = load_folder("data/folkrnn_both")
    output_meter = load_folder("data/folkrnn_meter")
    output_key = load_folder("data/folkrnn_key")

    output_modified = load_folder("data/modified")
    oneilljigs_labels = load_folder("data/ONeill")

    
    #["ONeill","ONeill_10","ONeill_20","ONeill_30","ONeill_40","ONeill_50","ONeill_60","ONeill_70","ONeill_80","ONeill_90","ONeill_100"] 


    #source_label_list = [output_modified, output_both, output_meter, output_key, oneilljigs_labels]
    #source_label_list = [output_meter, output_key, output_modified]

    #source_label_list = [output_both]
    source_label_list = [oneilljigs_labels]
    
    #target_label_list = [data_both, data_both, data_meter, data_key, oneilljigs_labels]
    #target_label_list = [data_meter, data_key, data_both]

    #target_label_list = [data_both]
    target_label_list = [oneilljigs_labels]
    
    #default_name_list = ["modified", "both", "meter", "key", "oneill"] #"oneill", 
    default_name_list = ["both"]
    #default_name_list = ["oneill"]

    if len(source_label_list) != len(target_label_list):
        raise ValueError("Source and target label lists must have the same length")
    


    #plotting.plot_key_meter_grid(data_dir="data/data_v2", output_name="key_meter_grid")

    for i in range(len(source_label_list)):
        print(f"Processing source label list {i}: {source_label_list[i]}")
        print(f"Processing target label list {i}: {target_label_list[i]}")

        source_labels = source_label_list[i]
        target_labels = target_label_list[i]
        #get
        both_labels = source_labels + [label for label in target_labels if label not in source_labels]
        # Load abc and wav formats of the data
        print("Loading data for labels:", both_labels)
        data = get_data(both_labels)
        #print(data)

        # Shorten label names
        source_labels = [label.split("/")[-1] for label in source_labels]
        target_labels = [label.split("/")[-1] for label in target_labels]
        both_labels = source_labels + [label for label in target_labels if label not in source_labels]

        data = embed_all(data, embeddings)

        if True:
            for embedding in embeddings:
                for label in target_labels:
                    if len(data[label][embedding]) < 400:
                        plotting.local_dist_grid(data, label, embedding)
        plotting.plot_data_dist(data, target_labels, default_name_list[i])
        #plotting.make_embedding_plots(data, both_labels, embeddings)

        #plotting.make_distance_plots(data, source_labels, target_labels, embeddings, methods)


        attribution_configs = [  #top_N, dist_threshold, top_Y, attribution_threshold
            [None, None, None, None],
            #[5, None, None, None],
            [10, None, None, None],
            #[4, None, None, None],
            #[None, 0.2, None, None],
            [None, 0.3, None, None]
            #[None, 0.4, None, None],
            #[None, None, 3, None]
        ]

        #plotting.make_attribution_plots(data, source_labels, target_labels, embeddings, methods, attribution_configs)
        
        if i == 0 or i == 1: 
            for label in source_labels:
                if len(data[label][embeddings[0]]) > 15:
                    plotting.get_most_similar(data, label, target_labels, 12, embedding=embeddings[0], method=methods[0], N=5, out_file="plots/distances/most_similar_{}.txt".format(label))
                    plotting.get_most_similar(data, label, target_labels, 12, embedding=embeddings[0], method=methods[0], N=5, out_file="plots/distances/most_similar_{}_dense.txt".format(label), dense=True)
            for label in target_labels:
                if len(data[label][embeddings[0]]) > 15:
                    plotting.get_most_similar(data, label, target_labels, 12, embedding=embeddings[0], method=methods[0], N=5, out_file="plots/distances/most_similar_{}.txt".format(label))
                    plotting.get_most_similar(data, label, target_labels, 12, embedding=embeddings[0], method=methods[0], N=5, out_file="plots/distances/most_similar_{}_dense.txt".format(label), dense=True)

        plotting.get_most_similar(data, source_labels[0], target_labels, 15, embedding=embeddings[0], method=methods[0], N=5)
    