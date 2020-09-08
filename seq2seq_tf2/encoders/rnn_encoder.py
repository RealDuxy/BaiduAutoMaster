import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        '''
        initialize the encoder
        :param vocab_size: size of vocabulary
        :param embedding_dim: the dims of embedding matrix == 256
        :param enc_units: the number of units in single encoder layer = 256
        :param batch_sz: how many samples in one step = 32
        :param embedding_matrix: embedding_matrix
        '''

        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        # self.enc_units = enc_units // 2

        # 定义Embedding层，加载预训练的词向量
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                   weights=[embedding_matrix],
                                                   trainable=False)
        # tf.keras.layers.GRU自动匹配cpu、gpu
        # 定义单向的GRU层
        self.gru = tf.keras.layers.GRU(units=self.enc_units  # dimensionality of the output space
                                           # Whether to return the last output in the output sequence, or the full
                                           # sequence.
                                           # This is required in Bidirectional
                                           , return_sequences=True
                                           # Whether to return the last state in addition to the output.
                                           # This is required in Bidirectional
                                           , return_state=True
                                           )
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        '''

        :param x: the input sequence, (batch_size, input_dim, sequence_length) = (8, 300, enc_units=256)
        :param hidden: the initial hidden state
        :return: output: sequence of output of every units
                state: hidden state of last unit
        '''
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
