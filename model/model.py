import tensorflow as tf
from layers import Encoder, EloRegression, ResultClassification


class Chessformer(tf.keras.Model):
    def __init__(self, num_layers, vocab_size, d_k, num_heads, dff,
                 dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(
                num_layers, vocab_size, d_k, num_heads, dff, dropout_rate)
        self.elo_regression_head = EloRegression()
        self.result_classification_head = ResultClassification()

    def call(self, moves):
        # moves.shape == (batch_num, sentence_length, d_k)
        moves = self.encoder(moves)
        # Average the embeddings for all tokens
        game_embedding = tf.reduce_mean(moves, axis=1) 
        elos = self.elo_regression_head(game_embedding)
        result = self.result_classification_head(game_embedding)
        return elos, result
