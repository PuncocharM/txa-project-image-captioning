import tensorflow as tf
import numpy as np
import time


class RNNLanguageModel:
    def __init__(self, num_steps, vocab_size, embedding_size, lstm_size, learning_rate, sess, checkpoint_path, saver_max_to_keep=10):
        self.num_steps = num_steps
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_size = lstm_size
        self.learning_rate = learning_rate
        self.sess = sess
        self.checkpoint_path = checkpoint_path
        self.saver_max_to_keep = saver_max_to_keep

        self.step = 0
        self.losses = []
    
    def build_rnn(self, input_, input_lengths, state_feed = None, mode='train'):
        # Embedding layer
        with tf.device("/cpu:0"):
            embedding = tf.get_variable('E', initializer=tf.truncated_normal([self.vocab_size, self.embedding_size]))
            embedded_input = tf.nn.embedding_lookup(embedding, input_)

        # RNN layer
        lstm = tf.contrib.rnn.BasicLSTMCell(self.lstm_size)
        if mode == 'train':
            outputs, _ = tf.nn.dynamic_rnn(lstm, embedded_input, dtype=tf.float32, sequence_length=input_lengths)
        else:
            state_tuple = tf.contrib.rnn.LSTMStateTuple(*tf.unstack(state_feed, axis=0))
            outputs, self.infer_state = tf.nn.dynamic_rnn(lstm, embedded_input, dtype=tf.float32, sequence_length=input_lengths, initial_state=state_tuple)

        # Projection and softmax
        softmax_w = tf.get_variable('W_softmax', initializer=tf.truncated_normal([self.lstm_size, self.vocab_size]), dtype=tf.float32)   # [lstm_size, V]
        softmax_b = tf.get_variable('b_softmax', initializer=tf.zeros([self.vocab_size]))   # [V]
        logits = tf.tensordot(outputs, softmax_w, axes=[[2],[0]]) + tf.reshape(softmax_b, [1,1,-1]) # [B, T, V]
        return logits
    
    def build_train_graph(self):
        self.train_input = tf.placeholder(tf.int32, [None, self.num_steps], name='train_input')
        self.train_input_lengths = tf.placeholder(tf.int32, [None], name='train_input_lengths')
        self.train_target  = tf.placeholder(tf.int32, [None, self.num_steps], name='train_target')
        with tf.variable_scope('root'):
            train_logits = self.build_rnn(self.train_input, self.train_input_lengths)
        self.train_loss = self.build_loss(train_logits, self.train_target, self.train_input_lengths)
            
    def build_eval_graph(self, reuse=True):
        self.eval_input = tf.placeholder(tf.int32, [None, self.num_steps], name='eval_input')
        self.eval_input_lengths = tf.placeholder(tf.int32, [None], name='eval_input_lengths')
        self.eval_target  = tf.placeholder(tf.int32, [None, self.num_steps], name='eval_target') 
        with tf.variable_scope('root', reuse=reuse):
            eval_logits = self.build_rnn(self.eval_input, self.eval_input_lengths)
        self.eval_loss = self.build_loss(eval_logits, self.eval_target, self.eval_input_lengths)
        
    def build_infer_graph(self, reuse=True):
        self.infer_input = tf.placeholder(tf.int32, [None, 1], name='infer_input')
        self.infer_input_lengths = tf.placeholder(tf.int32, [None], name='infer_input_lengths')
        self.state_feed = tf.placeholder(dtype=tf.float32, shape=[2, 1, self.lstm_size], name='state_feed')
        with tf.variable_scope('root', reuse=reuse):
            logits = self.build_rnn(self.infer_input, self.infer_input_lengths, self.state_feed, mode='inference')
            self.infer_probas = tf.nn.softmax(logits)
            
    def build_loss(self, logits, targets, input_lengths):
        ''' Crossentropy between probas and one-hot target = negative log-likelihood of a correct word. '''
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        mask = tf.sequence_mask(input_lengths, maxlen=self.num_steps, dtype=tf.float32)
        masked_l = tf.multiply(l, mask)
        loss = tf.reduce_sum(masked_l) / tf.reduce_sum(mask)
        return loss
    
    def attach_training_ops(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.train_loss)
        
    def build(self, model_type='train'):
        if model_type == 'train':
            self.build_train_graph()
            self.attach_training_ops()
            self.build_eval_graph()
        if model_type == 'infer':
            self.build_infer_graph(reuse=False)
        self.saver = tf.train.Saver(max_to_keep=self.saver_max_to_keep)

    def generate_batches(self, X, Y, X_lens, batch_size, include_last=False):
        num_batches = X.shape[0] // batch_size
        idx = np.random.permutation(X.shape[0])
        for i in range(num_batches):
            start = i*batch_size
            end = (i+1)*batch_size
            b_ids = idx[start:end]
            yield X[b_ids], Y[b_ids], X_lens[b_ids]
        if include_last:
            b_ids = idx[num_batches*batch_size:]
            if len(b_ids) > 0:
                yield X[b_ids], Y[b_ids], X_lens[b_ids]
            
    def evaluate(self, X, Y, X_lens):
        # evaluate batch-by-batch, because allocating large tensors fails on memory
        loss = 0.0
        i = 0
        for x, y, x_lens in self.generate_batches(X, Y, X_lens, batch_size=1000, include_last=True):
            loss += self.sess.run(self.eval_loss, feed_dict={
                self.eval_input: x,
                self.eval_input_lengths: x_lens,
                self.eval_target: y
            })
            i += 1
        return loss / i
        
    def train(self, X, Y, X_lens, n_epochs, batch_size, evaluate_every=10, save_every=10):
        t_a = time.clock()
        for epoch in range(n_epochs):
            print('epoch', epoch+1)
            for x, y, x_lens in self.generate_batches(X, Y, X_lens, batch_size):
                self.step += 1
                self.sess.run(self.train_op, feed_dict={
                    self.train_input: x, 
                    self.train_input_lengths: x_lens,
                    self.train_target: y
                })
                if self.step % evaluate_every == 0:
                    loss_value = self.evaluate(X, Y, X_lens)
                    self.losses.append(loss_value)
                    t_b = time.clock()
                    elapsed = time.strftime('%Hh %Mm %Ss', time.gmtime(t_b - t_a))
                    print('training loss after', self.step, 'steps:', loss_value, 'elapsed time:', elapsed)
                if self.step % save_every == 0:
                    saved_path = self.saver.save(self.sess, self.checkpoint_path, global_step=self.step)
                    np.save('losses', self.losses)
                    print('saved model to', saved_path)

        print('Finished training')
        saved_path = self.saver.save(self.sess, self.checkpoint_path + '-final')
        np.save('losses', self.losses)
        print('Saved final model to', saved_path)
        print('Final training loss:', self.evaluate(X, Y, X_lens))
        t_b = time.clock()
        print(time.strftime('It took %Hh %Mm %Ss', time.gmtime(t_b - t_a)))
        
    def infer(self, X, X_lens, state):
        return self.sess.run([self.infer_probas, self.infer_state], feed_dict={
            self.infer_input: X, #np.reshape(seq[-1], (1,1)),
            self.infer_input_lengths: X_lens, #np.array([1]),
            self.state_feed: state
        })