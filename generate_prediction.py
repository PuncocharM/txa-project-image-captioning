import tensorflow as tf
import numpy as np
import time
import pickle
import json
from pycocotools.coco import COCO
from config import *
from lstm_captioning_model import RNNCaptioningModel

print('Loading descriptors...')
xc_val = pickle.load(open(ms_coco_dir + '/descriptors/xception/xception-val2017-descriptors.pic', 'rb'))
descriptors_val = xc_val
print('... loading descriptors done.')

print('Loading vocabulary...')
with open('vocab-train2017.pic', 'rb') as f:
    vocab = pickle.load(f)
id_word = [v[0] for v in vocab]
word_id = {w:i for i,w in enumerate(id_word)}
vocab_size = len(vocab)
print('... loading vocabulary done.', vocab_size)

print('Building model...')
image_descriptor_size = 2048
tf.reset_default_graph()
sess = tf.Session()
model = RNNCaptioningModel(embedding_size=embeding_size, image_descriptor_size=image_descriptor_size, lstm_size=lstm_size, num_steps=num_steps, vocab_size=vocab_size, sess=sess, checkpoint_path=checkpoint_path)
model.build(model_type='infer')
model.saver.restore(sess, checkpoint_path + '-1000')
print('... building model done.')


cocoVal = COCO(ms_coco_dir + '/annotations/captions_val2017.json')
valImgIds = cocoVal.getImgIds()

batch_size = 1000
num_batches = (len(valImgIds) + batch_size - 1) // batch_size

predictions = []
t_start = time.clock()

for i in range(0, len(valImgIds), batch_size):
    print('{}/{}'.format(i // batch_size + 1, num_batches))
    imgs = []
    for j in range(i, i + batch_size):
        imgId = valImgIds[j]
        imgs.append(descriptors_val[imgId])
    imgs = np.stack(imgs)
    seqs = model.infer(imgs, start_id=word_id['START'], end_id=word_id['END'])
    for j, k in enumerate(range(i, i + batch_size)):
        imgId = valImgIds[k]
        caption = ' '.join([id_word[w] for w in seqs[j] if id_word[w] not in ('START', 'END')])
        caption = caption.replace(' ,', ',').replace(' .', '.')  # not the best way
        predictions.append({'image_id': imgId, 'caption': caption})

    t_now = time.clock()
    t_elapsed = t_now - t_start
    str_elapsed = time.strftime('%Hh %Mm %Ss', time.gmtime(t_elapsed))
    t_eta = t_elapsed / (i + 1) * (len(valImgIds) - i - 1)
    str_eta = time.strftime('%Hh %Mm %Ss', time.gmtime(t_eta))
    print('elapsed:', str_elapsed, '. ETA:', str_eta)

with open('predictions-lstm-1000.json', 'w') as f:
    json.dump(predictions, f)

print('Saved prediction.')