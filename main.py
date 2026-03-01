train_data_path = "folkrnn/data/ONeillsJigs_parsed_wot"
embedding = "clap"
import pysynth
import random
import librosa
import torch
from muq import MuQ, MuQMuLan
import laion_clap


def load_data(data_path):
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

    return tunes, idx2token, token2idx

def format_abc(tune):
    #print(tune)
    pass#TODO

def ABC2wav(tune):
    formatted_abc = format_abc(tune)
    #wav = pysynth.make_wav(formatted_abc) #TODO
    return "data/wav/cmajor_piano.wav" #TODO

def tunes_to_wav(tunes):
    #TODO check if the wav files already exist, if not, generate them using ABC2wav
    res = []
    for tune in tunes: 
        wav_fname = ABC2wav(tune)
        res.append(wav_fname)
    return res

def embed(tunes, tune_fnames, embedding):
    embedding = embedding.lower()
    if embedding == "clamp":
        return clamp(tunes)
    if embedding == "clap":
        return clap(tunes)
    if embedding == "muq":
        return muq(tunes)
    if embedding == "folkrnn":
        return folkrnn_embed(tunes)
    
def clamp(tunes):
    res = []
    for t in tunes:
        res.append([random.random() for _ in range(100)])
        pass#TODO
    return res

def clap(tune_fnames):
    res = []
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt() # download the default pretrained checkpoint.
    for wav_fname in tune_fnames:

        # Get audio embeddings from audio data
        audio_data, _ = librosa.load(wav_fname, sr=48000) # sample rate should be 48000
        audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
        audio_embed = model.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
        print("Audio embed first 20:", audio_embed[:,-20:])
        print("Audio embed shape:", audio_embed.shape)
        res.append(audio_embed[0])
    return res

def muq(tune_fnames):
    res = []
    device = 'cpu'
    mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    mulan = mulan.to(device).eval()
    for wav_fname in tune_fnames:
        wav, sr = librosa.load(wav_fname, sr = 24000)

        # Extract music embeddings
        wav, sr = librosa.load(wav_fname, sr = 24000)
        wavs = torch.tensor(wav).unsqueeze(0).to(device) 
        with torch.no_grad():
            audio_embeds = mulan(wavs = wavs) 
        print("audio embeds: ", audio_embeds, audio_embeds.shape)

        # Extract text embeddings (texts can be in English or Chinese)
        texts = ["classical genres, hopeful mood, piano.", "一首适合海边风景的小提琴曲，节奏欢快"]
        with torch.no_grad():
            text_embeds = mulan(texts = texts)

        # Calculate dot product similarity
        sim = mulan.calc_similarity(audio_embeds, text_embeds)
        print(sim)
        # Convert audio_embeds to list and append to res
        res.append(audio_embeds[0].cpu().numpy().tolist())
    return res

def folkrnn_embed(tunes):
    res = []
    for t in tunes:
        res.append([random.random() for _ in range(100)])
        pass#TODO
    return res



if __name__ == "__main__":
    tunes, idx2token, token2idx = load_data(train_data_path)
     
    print(tunes[0])
    print()
    print(idx2token)
    print()
    print(token2idx)


    tune_fnames = tunes_to_wav(tunes)
    print(tune_fnames)
    embedded_data = embed(tunes, tune_fnames, embedding)

    