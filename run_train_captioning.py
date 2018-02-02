import tensorflow as tf
import pickle
from lstm_captioning_model import RNNCaptioningModel
from utils import load_captions_and_images
from config import *


print('Loading descriptors...')
xc_train = pickle.load(open(ms_coco_dir + '/descriptors/xception/xception-train2017-descriptors.pic', 'rb'))
xc_val = pickle.load(open(ms_coco_dir + '/descriptors/xception/xception-val2017-descriptors.pic', 'rb'))

descriptors_train = xc_train
descriptors_val = xc_val
print('... loading descriptors done.')

print('Loading training data...')
X_images, X_captions, Y, X_lens, vocab_size, _ = load_captions_and_images(descriptors_train, data_type='train2017')
X_images_val, X_captions_val, Y_val, X_lens_val, _, _ = load_captions_and_images(descriptors_val, data_type='val2017')
print('...loading training data done.')


print('Building model...')
tf.reset_default_graph()
#gpu_fraction = 0.8
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
sess = tf.Session()

image_descriptor_size = X_images.shape[1]

model = RNNCaptioningModel(embedding_size=embeding_size, image_descriptor_size= image_descriptor_size, lstm_size=lstm_size, num_steps=num_steps, vocab_size=vocab_size, sess=sess, checkpoint_path=checkpoint_path)
model.build(model_type='train')

sess.run(tf.global_variables_initializer())
# model.saver.restore(sess, 'from-server/' + checkpoint_path + '-10')
print('...building model done.')


n_epochs = 10

model.train(X_images, X_captions, Y, X_lens, n_epochs, batch_size, learning_rate, save_every=10,
            X_imgs_val=X_images_val, X_captions_val=X_captions_val, Y_val=Y_val, X_lens_val=X_lens_val)

sess.close()