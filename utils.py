def load_captions(ms_coco_dir):
    """
    This is an old function used with language modelling, not by captioning anymore.
    See new functions below.
    """
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


def load_and_save_captions(ms_coco_dir, train_size, num_steps, freq_threshold, data_type='train2017', vocab=None):
    """
    Loads captions from ms_coco, performs tokenization, cropping, padding etc.
    Saves vocab and (X_captions, X_lens, Y, caption_ids) to .pic files so that tokenization can be done only once.

    :param ms_coco_dir:
    :param data_type: default 'train2017' or 'val2017'
    :param vocab: pass training vocab if val
    :return: None
    """
    import numpy as np
    from pycocotools.coco import COCO
    import nltk
    import itertools
    import pickle

    ann_file = '{}/annotations/captions_{}.json'.format(ms_coco_dir, data_type)
    coco = COCO(ann_file)

    imgIds = coco.getImgIds()
    annIds = coco.getAnnIds(imgIds=imgIds)
    anns = coco.loadAnns(annIds)
    captions_original = [a['caption'] for a in anns]
    caption_ids_original = [a['image_id'] for a in anns]
    print('# of images', len(imgIds), '# of captions', len(anns), '# of captions per image', len(anns) / len(imgIds))

    # limit training set to this number of first sentences
    captions = captions_original[:train_size]
    caption_ids = caption_ids_original[:train_size]

    print('Tokenization...')
    token_start = 'START'
    token_end = 'END'
    tok_captions = [[token_start] + [w for w in nltk.word_tokenize(caption) if w] + [token_end] for caption in captions]
    print('... tokenization done.')

    if vocab is None:
        print('Building vocabulary...')
        freq_dist = nltk.FreqDist(itertools.chain(*tok_captions))
        vocab = [(k, v) for k, v in freq_dist.items() if v >= freq_threshold]

        print('Saving vocab to vocab.pic')
        with open('vocab.pic', 'wb') as f:
            pickle.dump(vocab, f)

    vocab_size = len(vocab)
    print('vocab size', vocab_size)
    id_word = [v[0] for v in vocab]
    word_id = {w: i for i, w in enumerate(id_word)}
    print('... building vocabulary done.')

    print('Processing captions...')
    # drop all words not in vocabulary
    # maybe to replace them with special token would be better?
    tok_captions = [[w for w in s if w in word_id] for s in tok_captions]

    # cut all sentences to num_steps
    tok_captions = [s[:num_steps] for s in tok_captions]

    X_captions = [[word_id[w] for w in s[:-1]] for s in tok_captions]
    Y = [[word_id[w] for w in s[1:]] for s in tok_captions]  # shift-by-1 X, next-word prediction

    X_lens = np.asarray([len(x) for x in X_captions])

    # pad with zeros to num_steps
    X_captions = np.asarray([np.pad(x, (0, num_steps - len(x)), 'constant') for x in X_captions])
    Y = np.asarray([np.pad(y, (0, num_steps - len(y)), 'constant') for y in Y])
    print('... processing captions done.')

    print('Saving (X_captions, X_lens, Y, caption_ids) to XY.pic')
    with open('XY-{}.pic'.format(data_type), 'wb') as f:
        pickle.dump((X_captions, X_lens, Y, caption_ids), f)


def load_captions_and_images(descriptors, data_type='train2017'):
    """
    This loads the files generated by load_and_save_captions(...) and returns the prepared training set.

    :param descriptors: dict[img_id] -> descriptor
    :param data_type: default 'train2017' or 'val2017'
    :return: X_images, X_captions, Y, X_lens, vocab_size, caption_ids
    """
    import numpy as np
    import pickle

    with open('vocab.pic', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    with open('XY-{}.pic'.format(data_type), 'rb') as f:
        (X_captions, X_lens, Y, caption_ids) = pickle.load(f)

    X_images = np.stack([descriptors[img_id] for img_id in caption_ids])

    return X_images, X_captions, Y, X_lens, vocab_size, caption_ids