import pysynth
import random
import librosa
import torch
from muq import MuQ, MuQMuLan

wav, sr = librosa.load('data/wav/cmajor_piano.wav', sr = 24000)
device = 'cpu'
mulan = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
mulan = mulan.to(device).eval()

# Extract music embeddings
wav, sr = librosa.load('data/wav/cmajor_piano.wav', sr = 24000)
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