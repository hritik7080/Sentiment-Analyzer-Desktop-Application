import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from keras.utils.np_utils import to_categorical

# %%
yelp = pd.read_csv('./train.csv', encoding='ISO-8859-1')
yelp['SentimentText'] = yelp['SentimentText'].apply(lambda x: x.lower())
lm=WordNetLemmatizer()
yelp['SentimentText'] = yelp['SentimentText'].apply(lambda x: lm.lemmatize(x))
import re

yelp['SentimentText'] = yelp['SentimentText'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))


# %%
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(yelp['SentimentText'].values)
# %%
x = tokenizer.texts_to_sequences(yelp['SentimentText'].values)
# %%
x = pad_sequences(x)
# %%
embed_dim = 128
lstm_out = 196
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=x.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# %%

y = yelp['Sentiment'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
batch_size = 100
#%%
model.fit(x_train, y_train, epochs=100, batch_size=batch_size, verbose=2)
#%%
x = ['We are good']
print(x)
text = tokenizer.texts_to_sequences(x)
text = pad_sequences(text, maxlen=x_train.shape[1], dtype='int32', value=0)
print(text)
sentiment = model.predict(text, batch_size=1, verbose=2)
print(sentiment)
#%%
model.save('E:\\Int213\\Class work\\model.h5', )

#%%
