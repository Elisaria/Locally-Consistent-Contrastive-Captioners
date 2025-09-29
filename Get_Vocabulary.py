from keras import layers
import tensorflow as tf
import numpy as np


# Add "[cls]" token to each sentence.
def add_cls_token(line):
    punctuation_pattern = r"[!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~—·‘’“”《》【】（），。、；：？！…]"
    # Remove punctuation marks
    line = tf.strings.regex_replace(
        input=line,
        pattern=punctuation_pattern,
        rewrite=""
    )
    line_with_cls = tf.strings.join([line, "[cls]"], separator=" ")
    return line_with_cls


text_file = 'texts.txt'
texts = tf.data.TextLineDataset(text_file)
texts = texts.map(add_cls_token)

# Print the text line by line for viewing
for line in texts:
    print(line)

Vectorize_Layer = layers.TextVectorization(output_sequence_length=77)
# Learn a vocabulary list from the text dataset
Vectorize_Layer.adapt(texts.batch(64))
# Print the vocabulary list
print(Vectorize_Layer.get_vocabulary())

vocabulary = "vocabulary.txt"
tf.io.write_file(vocabulary, "\n".join(Vectorize_Layer.get_vocabulary()))

# Convert all texts in the text dataset into integer sequences
# (batch_size, 77)
text_sequences = Vectorize_Layer(tf.stack(list(texts)))
print(text_sequences)
# Save as an .npy file
np.save("texts.npy", text_sequences.numpy())
