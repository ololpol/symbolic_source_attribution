"""
This file modifies data files to a format in which they can be processed by abc2midi. This is used whem embedding using CLAP or MUQ.
Abc2midi can then be applied on the resultant file to generate a midi file, which in turn can be made into a wav file that the audio embeddings operate on.

The resulting file is stored in the same location, but has the "_labelled" ending added to its filename.

"""

#python abc_processing.py --data_path data/ONeill.abc

import argparse
parser = argparse.ArgumentParser(description='Process ABC notation data.')
parser.add_argument('--data_path', type=str, default='data/ONeill.abc', help='Path to the ABC notation data file.')
args = parser.parse_args()


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



target_file = args.data_path
tunes, _, _ = load_abc(target_file)


ending = target_file[-4:]
print("ending:", ending)
write_path = target_file[:-4] + "_labelled"+ending
with open(write_path, 'w') as f:
    i = 0
    for tune in tunes:
        tune = tune.replace(' ', '')

        i += 1
        f.write("X:" + str(i) + "\n")
        f.write(tune + "\n\n")