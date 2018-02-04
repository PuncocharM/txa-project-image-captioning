import tensorflow as tf
import time
from lstm_language_model import RNNLanguageModel
from utils import load_captions
from config import *

X, Y, X_lens, vocab_size = load_captions(ms_coco_dir)
print('done with data preparation')

tf.reset_default_graph()
#gpu_fraction = 0.8
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
sess = tf.Session()

model = RNNLanguageModel(embedding_size=embeding_size, learning_rate=learning_rate, lstm_size=lstm_size, num_steps=num_steps, vocab_size=vocab_size, sess=sess, checkpoint_path=checkpoint_path)
model.build(model_type='train')

sess.run(tf.global_variables_initializer())
# model.saver.restore(sess, 'from-server/' + checkpoint_path + '-10')

print('Done with building the model')

n_epochs = 10

t_a = time.clock()
model.train(X, Y, X_lens, n_epochs, batch_size, save_every=100)
t_b = time.clock()
print(time.strftime('It took %Hh %Mm %Ss', time.gmtime(t_b - t_a)))

sess.close()