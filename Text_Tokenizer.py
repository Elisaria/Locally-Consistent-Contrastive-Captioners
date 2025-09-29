from keras import layers
import tensorflow as tf

# The number of lines in the vocabulary.txt
vocabulary_size = 15


# The number of multi-head self-attention layers is 12,
# with 8 attention heads per layer,
# and the dimension of all latent layers is 512.
class Text_Tokenizer(layers.Layer):
    def __init__(self, model_dim=512, num_layers=12):
        super().__init__()
        self.num_layers = num_layers
        self.Text_Embedding = layers.Embedding(input_dim=vocabulary_size,
                                               output_dim=model_dim,
                                               mask_zero=True)
        self.MHAs = [layers.MultiHeadAttention(num_heads=8, key_dim=model_dim) for _ in range(num_layers)]

    # input : (B, L=77)
    # output : (B, L=77, 512)
    def call(self, inputs, *args, **kwargs):
        # (B, L=77, 512)
        x = self.Text_Embedding(inputs)
        for i in range(self.num_layers):
            x = self.MHAs[i](query=x,
                             value=x,
                             key=x)
        return x


# forward propagation
if __name__ == '__main__':
    inputs = tf.constant([[3, 8, 2, 0, 0]])
    print(Text_Tokenizer()(inputs))
