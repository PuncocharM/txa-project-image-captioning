def load_captions(ms_coco_dir):
    import numpy as np
    from pycocotools.coco import COCO
    import nltk
    import itertools
    import pickle
    from config import train_size, num_steps

    annFileTrain = '{}/annotations/captions_train2017.json'.format(ms_coco_dir)
    # annFileVal = '{}/annotations/captions_val2017.json'.format(ms_coco_dir)

    cocoTrain = COCO(annFileTrain)
    # cocoVal = COCO(annFileVal)

    imgIds = cocoTrain.getImgIds()
    annIds = cocoTrain.getAnnIds(imgIds=imgIds)
    anns = cocoTrain.loadAnns(annIds)
    captions_original = [a['caption'] for a in anns]
    print('# of imabes', len(imgIds), '# of captions', len(anns), '# of captions per image', len(anns) / len(imgIds))

    train_size = train_size  # limit training set to this number of first sentences
    captions = captions_original[:train_size]

    token_start = 'START'
    token_end = 'END'
    tok_captions = [[token_start] + [w for w in nltk.word_tokenize(caption) if w] + [token_end] for caption in captions]

    freq_dist = nltk.FreqDist(itertools.chain(*tok_captions))

    freq_threshold = 0
    vocab = [(k, v) for k, v in freq_dist.items() if v >= freq_threshold]
    vocab_size = len(vocab)
    id_word = [v[0] for v in vocab]
    word_id = {w: i for i, w in enumerate(id_word)}
    print('vocab size', vocab_size)

    with open('vocab.pic', 'wb') as f:
        pickle.dump(vocab, f)

    # drop all words not in vocabulary (less frequent than freq_threshold)
    # maybe to replace them with special token would be better?
    # now we actually don't drop anything
    tok_captions = [[w for w in s if w in word_id] for s in tok_captions]


    # cut all sentences to max_len
    tok_captions = [s[:num_steps] for s in tok_captions]

    X = [[word_id[w] for w in s[:-1]] for s in tok_captions]
    Y = [[word_id[w] for w in s[1:]] for s in tok_captions]  # shift-by-1 X, next-word prediction

    X_lens = np.asarray([len(x) for x in X])

    # pad with zeros to max_len
    X = np.asarray([np.pad(x, (0, num_steps - len(x)), 'constant') for x in X])
    Y = np.asarray([np.pad(y, (0, num_steps - len(y)), 'constant') for y in Y])

    return X, Y, X_lens, vocab_size