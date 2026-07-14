import os
def load_abc(data_path, folder = None):
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

def load_data(file_path):
    tunes, _, _ = load_abc(file_path)
    meters = []
    keys = []
    data = []
    for tune in tunes:
        parts = tune.split('\n')
        meters.append(parts[0])
        keys.append(parts[1])
        data.append(parts[2])
    

    return meters, keys, data


def set_key(keys, target):
    copy = [0] * len(keys)
    for i in range(len(keys)):
        copy[i] = target
    return copy

def set_meter(meters, target):
    copy = [0] * len(meters)
    for i in range(len(meters)):
        copy[i] = target
    return copy


def write_abc(meters, keys, data, file_path):
    if len(meters) != len(keys) or len(keys) != len(data):
        raise ValueError("Length of meters, keys, and data must be the same.")
    tunes = []
    for i in range(len(meters)):
        tune = f"{meters[i]}\n{keys[i]}\n{data[i]}"
        tunes.append(tune)
    abc_content = '\n\n'.join(tunes)
    with open(file_path, 'w') as f:
        f.write(abc_content)


source_files = ['data/folkrnn_both/folkrnn_v2-Cmaj-6-8.txt', 'data/folkrnn_both/folkrnn_v2-Cmaj-4-4.txt']
target_keys = ["Cmaj", "Cmin", "Cdor", "Cmix"]
target_meters = ["2/4", "3/2", "3/4", "4/4", "6/8", "9/8", "12/8"]

folder = "data/modified/"
if not os.path.exists(folder):
    os.makedirs(folder)
for file in source_files:
    meters, keys, data = load_data(file)
    id = keys[0] + "_" + meters[0].replace("/","-")
    for key in target_keys:
        keys_p = set_key(keys, "K:"+key)
        for meter in target_meters:
            meters_p = set_meter(meters, "M:"+meter)
            id_new = key + "_" + meter.replace("/","-")

            write_abc(meters_p, keys_p, data, folder+id+"_to_"+id_new+".txt")
