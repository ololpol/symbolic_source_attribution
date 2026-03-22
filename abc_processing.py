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



target_file = "data/allabcdodec_parsed_wot_0"
tunes, _, _ = load_abc(target_file)



with open(target_file + "_labelled", 'w') as f:
    i = 0
    for tune in tunes:
        tune = tune.replace(' ', '')

        i += 1
        f.write("X:" + str(i) + "\n")
        f.write(tune + "\n\n")