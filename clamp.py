from clamp_utils import *
from transformers import AutoTokenizer

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



# parse arguments
CLAMP_MODEL_NAME = 'sander-wood/clamp-small-512'
QUERY_MODAL = "music"

# load CLaMP model
model = CLaMP.from_pretrained(CLAMP_MODEL_NAME)

print(model.config)
music_length = 1024 #model.config.max_length
model = model.to(device)
model.eval()

# initialize patchilizer, and softmax
patchilizer = MusicPatchilizer()
softmax = torch.nn.Softmax(dim=1)




if __name__ == "__main__":
    # load query
    if QUERY_MODAL=="music":
        query = load_music(filename = "music_query.abc")
    query = unidecode(query)


    # encode query
    query_ids = encoding_data([query], QUERY_MODAL, patchilizer, music_length)
    query_feature = get_features(query_ids, QUERY_MODAL, model, device)

    print(query_feature.shape)
    print(query_feature[:20])