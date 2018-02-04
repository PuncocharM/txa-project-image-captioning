import tensorflow as tf
import numpy as np
import time


class RNNCaptioningModel:
    def __init__(self, num_steps, vocab_size, image_descriptor_size, embedding_size, lstm_size, sess, checkpoint_path, dropout=0.0, saver_max_to_keep=10):
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.image_descriptor_size = image_descriptor_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.sess = sess
        self.checkpoint_path = checkpoint_path
        self.dropout = dropout
        self.saver_max_to_keep = saver_max_to_keep

        self.step = 0
        self.losses = []
        self.losses_x = []
        self.losses_val = []
        self.losses_val_x = []

    def build_rnn(self, img_input, text_input, input_lengths, state_feed=None, mode='train'):
        # Text Embedding
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('E', initializer=tf.truncated_normal([self.vocab_size, self.embedding_size]))
            embedded_text = tf.nn.embedding_lookup(embedding, text_input)

        # Image embedding
        img_embedding_w = tf.get_variable('IE_W', initializer=tf.truncated_normal(
            [self.image_descriptor_size, self.embedding_size], dtype=tf.float32))
        img_embedding_b = tf.get_variable('IE_B', initializer=tf.truncated_normal([self.embedding_size]))
        embedded_image = tf.nn.relu(tf.nn.xw_plus_b(img_input, img_embedding_w, img_embedding_b))

        # RNN layer
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        if mode == 'train' and self.dropout > 0:
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=self.dropout, output_keep_prob=self.dropout)

        with tf.variable_scope('lstm') as lstm_scope:
            zero_state = lstm.zero_state(batch_size=tf.shape(embedded_image)[0], dtype=tf.float32)
            _, initial_state = lstm(embedded_image, zero_state)
            self.initial_state = initial_state

            if mode == 'train':
                outputs, _ = tf.nn.dynamic_rnn(lstm, embedded_text, dtype=tf.float32, sequence_length=input_lengths,
                                               initial_state=initial_state, scope=lstm_scope)
            else:
                state_tuple = tf.contrib.rnn.LSTMStateTuple(*tf.unstack(state_feed, axis=0))
                outputs, self.infer_state = tf.nn.dynamic_rnn(lstm, embedded_text, dtype=tf.float32,
                                                              sequence_length=input_lengths, initial_state=state_tuple,
                                                              scope=lstm_scope)

        # Projection and softmax
        softmax_w = tf.get_variable('W_softmax', initializer=tf.truncated_normal([self.lstm_size, self.vocab_size]),
                                    dtype=tf.float32)
        softmax_b = tf.get_variable('b_softmax', initializer=tf.zeros([self.vocab_size]))
        logits = tf.tensordot(outputs, softmax_w, axes=[[2], [0]]) + tf.reshape(softmax_b, [1, 1, -1])
        return logits

    def build_train_graph(self):
        self.train_input_images = tf.placeholder(tf.float32, [None, self.image_descriptor_size],
                                                 name='train_input_image')
        self.train_input = tf.placeholder(tf.int32, [None, self.num_steps], name='train_input')
        self.train_input_lengths = tf.placeholder(tf.int32, [None], name='train_input_lengths')
        self.train_target = tf.placeholder(tf.int32, [None, self.num_steps], name='train_target')
        with tf.variable_scope('root'):
            train_logits = self.build_rnn(self.train_input_images, self.train_input, self.train_input_lengths)
        self.train_loss = self.build_loss(train_logits, self.train_target, self.train_input_lengths)

    def build_eval_graph(self, reuse=True):
        self.eval_input_images = tf.placeholder(tf.float32, [None, self.image_descriptor_size], name='eval_input_image')
        self.eval_input = tf.placeholder(tf.int32, [None, self.num_steps], name='eval_input')
        self.eval_input_lengths = tf.placeholder(tf.int32, [None], name='eval_input_lengths')
        self.eval_target = tf.placeholder(tf.int32, [None, self.num_steps], name='eval_target')
        with tf.variable_scope('root', reuse=reuse):
            eval_logits = self.build_rnn(self.eval_input_images, self.eval_input, self.eval_input_lengths)
        self.eval_loss = self.build_loss(eval_logits, self.eval_target, self.eval_input_lengths)

    def build_infer_graph(self, reuse=True):
        self.infer_input_images = tf.placeholder(tf.float32, [None, self.image_descriptor_size],
                                                 name='infer_input_image')
        self.infer_input = tf.placeholder(tf.int32, [None, 1], name='infer_input')
        self.infer_input_lengths = tf.placeholder(tf.int32, [None], name='infer_input_lengths')
        self.state_feed = tf.placeholder(dtype=tf.float32, shape=[2, None, self.lstm_size], name='state_feed')
        with tf.variable_scope('root', reuse=reuse):
            logits = self.build_rnn(self.infer_input_images, self.infer_input, self.infer_input_lengths,
                                    self.state_feed, mode='inference')
            logits = tf.squeeze(logits, axis=1)
            self.infer_probas = tf.nn.softmax(logits)

    def build_loss(self, logits, targets, input_lengths):
        ''' Crossentropy between probas and one-hot target = negative log-likelihood of a correct word. '''
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        mask = tf.sequence_mask(input_lengths, maxlen=self.num_steps, dtype=tf.float32)
        masked_l = tf.multiply(l, mask)
        loss = tf.reduce_sum(masked_l) / tf.reduce_sum(mask)
        return loss

    def attach_training_ops(self):
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.train_loss)

    def build(self, model_type='train'):
        if model_type == 'train':
            self.build_train_graph()
            self.attach_training_ops()
            self.build_eval_graph()
        if model_type == 'infer':
            self.build_infer_graph(reuse=False)
        self.saver = tf.train.Saver(max_to_keep=self.saver_max_to_keep)

    def generate_batches(self, X_imgs, X_captions, Y, X_lens, batch_size, include_last=False):
        num_batches = X_imgs.shape[0] // batch_size
        idx = np.random.permutation(X_imgs.shape[0])
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            b_ids = idx[start:end]
            yield X_imgs[b_ids], X_captions[b_ids], Y[b_ids], X_lens[b_ids]
        if include_last:
            b_ids = idx[num_batches * batch_size:]
            if len(b_ids) > 0:
                yield X_imgs[b_ids], X_captions[b_ids], Y[b_ids], X_lens[b_ids]

    def evaluate(self, X_imgs, X_captions, Y, X_lens):
        loss = 0.0
        i = 0
        for x_imgs, x_captions, y, x_lens in self.generate_batches(X_imgs, X_captions, Y, X_lens, batch_size=1000,
                                                                   include_last=True):
            loss += self.sess.run(self.eval_loss, feed_dict={
                self.eval_input_images: x_imgs,
                self.eval_input: x_captions,
                self.eval_input_lengths: x_lens,
                self.eval_target: y
            })
            i += 1
        return loss / i

    def train(self, X_imgs, X_captions, Y, X_lens, n_epochs, batch_size, learning_rate, evaluate_every=10,
              save_every=10, X_imgs_val=None, X_captions_val=None, Y_val=None, X_lens_val=None):
        t_a = time.clock()
        for epoch in range(n_epochs):
            print('epoch', epoch + 1)
            for x_imgs, x_captions, y, x_lens in self.generate_batches(X_imgs, X_captions, Y, X_lens, batch_size):
                self.step += 1
                _, loss_value = self.sess.run([self.train_op, self.train_loss], feed_dict={
                    self.train_input_images: x_imgs,
                    self.train_input: x_captions,
                    self.train_input_lengths: x_lens,
                    self.train_target: y,
                    self.learning_rate: learning_rate
                })
                self.losses.append(loss_value)
                self.losses_x.append(self.step)
                if self.step % evaluate_every == 0:
                    print('Step', self.step)
                    t_b = time.clock()
                    elapsed = time.strftime('%Hh %Mm %Ss', time.gmtime(t_b - t_a))
                    print('  elapsed', elapsed)
                    # loss_value = self.evaluate(X_imgs, X_captions, Y, X_lens)
                    print('  training loss: {:.4f}'.format(loss_value))
                    if X_imgs_val is not None:
                        loss_value_val = self.evaluate(X_imgs_val, X_captions_val, Y_val, X_lens_val)
                        self.losses_val.append(loss_value_val)
                        self.losses_val_x.append(self.step)
                        print('  validation loss: {:.4f}'.format(loss_value_val))
                if self.step % save_every == 0:
                    saved_path = self.saver.save(self.sess, self.checkpoint_path, global_step=self.step)
                    np.save('losses', [self.losses, self.losses_x, self.losses_val, self.losses_val_x])
                    print('saved model to', saved_path)

        print('Finished training')
        saved_path = self.saver.save(self.sess, self.checkpoint_path + '-final')
        np.save('losses', [self.losses, self.losses_x, self.losses_val, self.losses_val_x])
        print('Saved final model to', saved_path)
        print('Final training loss:', self.evaluate(X_imgs, X_captions, Y, X_lens))
        t_b = time.clock()
        print(time.strftime('It took %Hh %Mm %Ss', time.gmtime(t_b - t_a)))

    def infer_seed_with_image(self, X_images):
        return self.sess.run(self.initial_state, feed_dict={
            self.infer_input_images: X_images,
        })

    def infer_step(self, X_captions, X_lens, state):
        return self.sess.run([self.infer_probas, self.infer_state], feed_dict={
            self.infer_input: X_captions,
            self.infer_input_lengths: X_lens,
            self.state_feed: state
        })

    def infer(self, img_descriptors, start_id, end_id, limit=50):
        if img_descriptors.ndim == 1:
            img_descriptors = img_descriptors.reshape((-1,1))
        batch_size = img_descriptors.shape[0]
        states = self.infer_seed_with_image(img_descriptors)
        seqs = np.zeros(shape=(limit+1, batch_size), dtype=np.int32)
        seqs[0, :] = start_id
        for i in range(limit):
            p, states = self.infer_step(np.reshape(seqs[i, :], (-1, 1)), np.ones(batch_size), states)
            for j in range(batch_size):
                #     max_id = np.argmax(p[j])
                max_id = np.random.choice(len(p[j]), p=p[j])
                seqs[i+1][j] = max_id
        seqs = seqs.T

        # after first END put all ENDs
        for i in range(batch_size):
            for j in range(limit):
                if seqs[i,j] == end_id:
                    seqs[i,j:] = end_id
                    break
        return seqs