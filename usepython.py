import tensorflow as tf
tf.config.list_physical_devices()

import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import re
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression


full_scryfall_df = pd.DataFrame(json.load(open("scryfall.json")))

df = full_scryfall_df[['name', 'mana_cost', 'cmc', 'type_line', 'oracle_text', 'power', 'toughness', 'colors', 'color_identity', 'keywords', 'set', 'released_at', 'rarity', 'games', 'legalities']]

# remove cards with the same name
df = df.sort_values(by=['released_at', 'name'])
df = df.drop_duplicates(subset=['name'])
df = df[df['games'].apply(lambda i: 'paper' in i)]

def legal(legalities):
    v = legalities.values()
    if len(set(v)) == 1 and "not_legal" in v:
        return False
    return True

df = df[df['legalities'].apply(legal)]

unsets = ['unglued', 'unhinged', 'unstable', 'unsanctioned', 'unfinity']
sets = json.loads(requests.get("https://api.scryfall.com/sets").text)
for s in sets["data"]:
    if s['name'].lower() in unsets:
        df = df[~df["set"].str.contains(s['code'])]

df.head()


df = df[~df["type_line"].str.contains("Token", na=False)] # remove tokens


# just vanilla creatures
vanilla_df = df[df["oracle_text"] == ""]
vanilla_df = vanilla_df[vanilla_df["type_line"].str.contains("Creature")]
vanilla_df


vanilla_df['cmc'].value_counts()




x = set()
for i, row in df.iterrows():
    if row['set'] not in ['afr', '40k', 'clb', 'sld']:
        x = x.union(row['keywords'])
    else:
        # ignore multimodal keywords
        x = x.union([i for i in row['keywords'] if ' ' not in i])
keyword_soup = ' '.join(x).lower()
print(keyword_soup)

removes = [r'\(.*?\)', r'\{.*?\}', r'—[^ ][^\n]*', r'(P|p)rotection(?! F)[^\n]*', r'\d*', r'Prototype[^\n]*']
#r'—[^{][^T][^\n]*', 
# todo protection, ward, a lot of stuff actually
def is_french_vanilla(row):
    text = row['oracle_text']
    
    if text == '': 
        return True # is just vanilla
    
    if text is np.nan:
        return False # is not valid
    
    for r in removes:
        text = re.sub(r, '', text)
    text = text.replace(',', '').replace(';', '')
    text = text.lower()
    text = text.strip()

    for i in text.split():
        if i not in keyword_soup:
            return False

    return True

french_vanilla_df = df[df["type_line"].str.contains("Creature", na=False)] # for now just creatures
french_vanilla_df = french_vanilla_df.sort_values(by=['name'])
french_vanilla_df['is_french_vanilla'] = french_vanilla_df.apply(is_french_vanilla, axis=1)
french_vanilla_df = french_vanilla_df[french_vanilla_df['is_french_vanilla']]

french_vanilla_df = french_vanilla_df.sort_values(by=['name'])

french_vanilla_df


french_vanilla_df[french_vanilla_df['rarity'] == 'mythic']

# todo organize code
alpha_release_date = dt.datetime(1993, 8, 5)
# ignore type line for now
data_df = french_vanilla_df.drop(columns=['name', 'mana_cost', 'type_line', 'oracle_text', 'color_identity', 'set', 'is_french_vanilla', 'games', 'legalities'])
data_df['cmc'] = data_df['cmc'].apply(int)
data_df['power'] = data_df['power'].apply(int)
data_df['toughness'] = data_df['toughness'].apply(int)
data_df['released_at'] = data_df['released_at'].apply(lambda i: (dt.datetime.strptime(i, '%Y-%m-%d').year - alpha_release_date.year))
data_df


def dummy_list(data_df, one_hot_df, column):
    x = set(data_df.explode(column)[column].values)
    x.remove(np.nan)
    
    for i in x:
        one_hot_df[f'{column}_{i}'] = data_df[column].apply(lambda j: int(i in j))
    

one_hot_df = pd.get_dummies(data_df.drop(columns=['colors', 'keywords']))
dummy_list(data_df, one_hot_df, 'colors')
dummy_list(data_df, one_hot_df, 'keywords')
one_hot_df

# make new dummy matrices for interaction terms
X = one_hot_df.drop(columns=['cmc', 'rarity_common'])
y = one_hot_df['cmc']

# make new linear regression
reg2 = LinearRegression().fit(X, y)

# print all coefficients
for name, coef in zip(X.columns, reg2.coef_):
    print(f"{name} Coefficient: {coef}")
print("Intercept:", reg2.intercept_)


# import spacy
# nlp = spacy.load("en_core_web_sm")
creature_df = df#[df["type_line"].str.contains("Creature", na=False)]
creature_df["mana_cost_spaced"] = creature_df["mana_cost"].apply(lambda i: i.replace("}{", "} {") if isinstance(i, str) else i)

tokens = {'': 0}
num_tokens = 1

def get_data(row):
    t = row["oracle_text"]
    g = t if t is np.nan else t.replace(row["name"], "{CARDNAME}").replace("\n", " {NEWLINE} ").replace(",", " {COMMA} ").replace(":", " {COLON} ").replace(".", " {PERIOD} ").replace(";", " {SEMICOLON} ").replace("}{", "} {")
    return [g]
#     return [row["name"], row["mana_cost_spaced"], row["type_line"], row["power"], row["toughness"], row["oracle_text"]]

def process(text):
    # remove reminder text & ,.
    return re.sub(removes[0], '', text).lower().replace(',', '').replace('.', '')

for i, row in creature_df.iterrows():
    for data in get_data(row):
#         doc = row["oracle_text"] # nlp().tokens took too long but im still considering it
        if not isinstance(data, str):
            continue # split cards have no orcale text
        data = process(data)

        for token in data.split():
            if token not in tokens:
                tokens[token] = num_tokens
                num_tokens += 1


print(num_tokens)
tokens_list = list(tokens.keys())

def encode(text):
    if text is np.nan:
        return []
    t = process(text).split()
    return [tokens[i] for i in t]

def encode_row(row):
    data = get_data(row)
    # todo pad mana cost & type line
    out = []
    for i in data:
        out.extend(encode(i))

    return out

def decode(encoded):
    return " ".join(tokens_list[i] for i in encoded)

name = "Pixie Illusionist"
t = creature_df[creature_df["name"] == name].iloc[0]
print(t)
print(encode_row(t))
print(decode(encode_row(t)))

creature_df["encoded"] = creature_df.apply(encode_row, axis=1)
creature_df["encoded_length"] = creature_df["encoded"].apply(len)
creature_df = creature_df[creature_df["encoded_length"] < 20]

total_words = max(creature_df["encoded_length"])
creature_df[creature_df["encoded_length"] == max(creature_df["encoded_length"])]

input_sequences = []
for seq in creature_df["encoded"]:
    for i in range(1, len(seq)):
        input_sequences.append(seq[:i + 1])



import tensorflow.keras.utils as ku
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# create predictors and label
total_words = num_tokens
predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
label = ku.to_categorical(label, num_classes=total_words)

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import tensorflow as tf

# https://github.com/nicknochnack/GANBasics/blob/main/FashionGAN-Tutorial.ipynb
# https://towardsdatascience.com/training-neural-networks-to-create-text-like-a-human-23bfdc23c28

model = Sequential()
model.add(Embedding(total_words, 240, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(150, return_sequences = True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


max(input_sequences[:,-1])

len(label[0])
total_words
import os
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=3497)

model.save_weights(checkpoint_path.format(epoch=0))
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.93):
            print("\nReached 93% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()
history = model.fit(predictors, label, epochs=300, verbose=1, callbacks=[cp_callback])

model.save_weights("completed_model.ckpt")
