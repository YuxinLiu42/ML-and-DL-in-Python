# 1. Different kinds of NNs
# 2. RNNs
# load required modules
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
# define model architecture
model = Sequential()
model.add(SimpleRNN(units = 64, input_shape = (None, 1)))
model.add(Dense(1, activation='sigmoid'))

# 3. The Embedding Layer
import numpy as np
# Define 10 restaurant reviews
reviews =['Never coming back!', 'horrible service',
          'rude waitress', 'cold food', 'horrible food!',
          'awesome', 'awesome services!', 'rocks',
          'poor work', 'couldn\'t have done better']
labels = np.array([1,1,1,1,1,0,0,0,0,0])

from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(reviews)
vocab_size = len(tokenizer.word_index) + 1 # bc unknown words
x_train = tokenizer.texts_to_sequences(reviews)
print(x_train)

from keras_preprocessing.sequence import pad_sequences
# define maximum length for the sequences
maxlen = 5
# .. and "pad" all samples accordingly
x_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
print(x_train)

from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
model = Sequential()
model.add(Embedding(input_dim = vocab_size,
                    output_dim = 8,
input_length = maxlen)) model.add(LSTM(units = 8)) #you don't know this yet ;-)
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['acc'])

# 4. LSTM / GRU
# load required modules
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
# define model architecture
model = Sequential()
model.add(Embedding(input_dim = 30000, output_dim = 128,
                    input_length = 256))
model.add(LSTM(units = 64, ativation = "tanh"))
model.add(Dense(1, activation='sigmoid'))

# 5. Bidirectionality
# load required modules
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Bidirectional, Dense
# define model architecture
model = Sequential()
model.add(Embedding(input_dim = 30000, output_dim = 128,
                    input_length = 256))
model.add(Bidirectional(LSTM(units = 64, ativation = "tanh")))
model.add(Dense(1, activation='sigmoid'))

# 6. Encoder-Decoder
# load required modules
from keras.models import Model
from keras.layers import Input, LSTM, Dense
# define encoder
encoder_inputs = Input(shape=(None, 256))
encoder = LSTM(units = 64, return_state = True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# define decoder
decoder_inputs = Input(shape=(None, 256))
decoder = LSTM(units = 64,
               return_sequences = True,
               return_state = True)
decoder_outputs, _, _ = decoder(decoder_inputs,
                                initial_state = encoder_states)
decoder_dense = Dense(256, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)