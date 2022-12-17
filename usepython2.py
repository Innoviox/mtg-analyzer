# An Analysis of Magic: The Gathering's Creatures

import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import re
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression

## Data Processing

#The data we will use is from [scryfall](https://scryfall.com/), a community-ran magic the gathering site. Scryfall's data is often more reliable and accurate than the data that Wizards produces, and it is also freely available for download through their API. This data contains every single card in Magic's history that was printed in English - 78,242 card objects. However, many of these objects are extraneous and would hurt our data analysis. For the next section, I will prune down these cards to exclude reprinted cards, illegal cards, joke cards, and many other types of cards that have been created over the years.

URL = "https://data.scryfall.io/default-cards/default-cards-20221212220657.json"
full_scryfall_df = pd.DataFrame(json.loads(requests.get(URL).text))
full_scryfall_df.head()

#This dataframe also comes with 84 columns, many of which we do not need. I will remove all columns except for the useful ones in determining a card's qualities, and in determining whether we want to analyze the card or not.

df = full_scryfall_df[['name',                       # the name of the card - not technically necessary but helpful for debugging
                       'mana_cost',                  # what type of mana the card costs to summon
                       'cmc',                        # how much mana the card costs
                       'type_line',                  # the type of the card (creature, sorcery, etc)
                       'oracle_text',                # what the card does
                       'power', 'toughness',         # the strength of the card if it's a creature
                       'colors', 'color_identity',   # more info on what type of mana the card costs
                       'keywords',                   # the keywords on the card (more on this later)
                       'set', 'released_at',         # when the card was released
                       'rarity',                     # how much the card was printed
                       'games',                      # games tells if it is legal online or in paper (we exclude online-only cards)
                       'legalities']]                # which formats the card is legal in

#First, we will remove all cards that are in there multiple times (e.g. they were printed in multiple sets). Wizards does this sometimes to bring back fan favorite cards or to have some basic cards that always work well. 

df = df.sort_values(by=['released_at', 'name'])
df = df.drop_duplicates(subset=['name'])

#Next, we will remove online-only cards. Wizards of the Coast released a program called Magic Arena, and to promote it they released cards that were only legal for that program. However, these cards were not created with the balance of the paper format in mind, and reference random effects and things only possible online. Therefore, I am excluding them from this analysis.

df = df[df['games'].apply(lambda i: 'paper' in i)]

#Some cards are illegal to play for power-level reasons (too strong for the format); however, we can still analyze these. The "not legal" designation means cards that are literally unplayable: they are printed alongside magic cards, but just say promotional text or act as other game pieces. Tokens are one such piece; some cards create tokens, but you can't put the actual token cards in your deck. However, Scryfall treats all of these as "card objects" and puts them in.

def legal(legalities):
    v = legalities.values()
    if len(set(v)) == 1 and "not_legal" in v:
        return False
    return True

df = df[df['legalities'].apply(legal)]
df = df[~df["type_line"].str.contains("Token", na=False)] # remove tokens

#Finally, some cards were designed as jokes by the Wizards designers in sets called "unsets". These cards, like the online cards, aren't tuned for interacting with any other cards, and so I will exclude them from this dataset.

unsets = ['unglued', 'unhinged', 'unstable', 'unsanctioned', 'unfinity']
sets = json.loads(requests.get("https://api.scryfall.com/sets").text)
for s in sets["data"]:
    if s['name'].lower() in unsets:
        df = df[~df["set"].str.contains(s['code'])]



df["num_colored_pips"] = df["mana_cost"].apply(lambda mana_cost: len(re.findall("\{[^\d]\}", str(mana_cost))))

#Let's take a look!

df.head()

#As you can see, we've now got the important parts of a card, and the cards are sorted conveniently by their release date. We are now looking at the first magic cards ever released. How strong were they? Let's find out!

## Vanilla Creature Analysis
#A vanilla creature is a creature with no text whatsoever - just stats! The "classic" vanilla creature is the Grizzly Bears, a 2/2 for 2 mana in green.

df[df["name"].str.contains("Grizzly Bears")]

#However, not all creatures are created equally. For example, the Coral Eel has the same mana cost, but only 1 toughness. The difference? It's in blue!

df[df["name"].str.contains("Coral Eel")]

#Clearly, some colors are better at producing creatures than other colors. But how much better? Let's start with vanilla creatures, since we know none of their abilities is influencing their mana cost. Therefore, we can just get a look at how much each point of power and toughness is costing, mana-wise.

vanilla_df = df[df["oracle_text"] == ""]
vanilla_df = vanilla_df[vanilla_df["type_line"].str.contains("Creature")]

for i in ['cmc', 'power', 'toughness']:
    vanilla_df[i] = vanilla_df[i].apply(int)

vanilla_df.head()

#vanilla_df.plot.bar('cmc')

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

def extract_ability_text(row):
    text = row['oracle_text']

    if text is np.nan:
        return ''
    
    for r in removes:
        text = re.sub(r, '', text)
    text = text.replace(',', '').replace(';', '')
    text = text.lower()
    text = text.strip()

    return text

def is_french_vanilla(row):
    text = row['oracle_text']
    
    if text == '': 
        return True # is just vanilla
    
    if text is np.nan:
        return False # is not valid
    
    text = extract_ability_text(row)

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


activated_abilities_df = df[df["type_line"].str.contains("Creature", na=False)]

def extract_activated_abilities(row):
    text = row['oracle_text']
    
    if text == '': 
        return [] # is just vanilla
    
    if text is np.nan:
        return [] # is not valid
    
    text = re.sub(removes[0], '', text)
    ret = []
    for i in text.split("\n"):
        if ":" in i:
#         if i.lower().split()[0].strip(",;") not in keyword_soup:
            ret.append(i.replace(row['name'], 'CARDNAME'))
    return ret
    
import collections
count = collections.defaultdict(int)
for i, row in activated_abilities_df.iterrows():
    ret = extract_activated_abilities(row)
    for j in ret:
        count[j] += 1

for k, v in count.items():
    if v > 5:
        print(k, v)


# import spacy
# nlp = spacy.load("en_core_web_sm")
creature_df = df[df["type_line"].str.contains("Creature", na=False)]
creature_df["mana_cost_spaced"] = creature_df["mana_cost"].apply(lambda i: i.replace("}{", "} {") if isinstance(i, str) else i)

tokens = {'': 0}
num_tokens = 1

replaces = [
      [",", " {comma} "],
      [":", " {colon} "],
      [".", " {period} "],
      [";", " {semicolon} "],
    ["\n", " {newline} "],
      ["}{", "} {"]]

def get_data(row):
    t = row["oracle_text"]
    if t is not np.nan:
        t = t.replace(row["name"], "{CARDNAME}")
        for i, j in replaces:
            t = t.replace(i, j)
    return [row["mana_cost_spaced"], row["type_line"], row["power"], row["toughness"], t]
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

creature_df.head()


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
    s = " ".join(tokens_list[i] for i in encoded)
    for i, j in replaces[:-1]:
        s = s.replace(j, i + " ")
        s = s.replace(j.strip(), i + " ")
    return s

name = "Pixie Illusionist"
t = creature_df[creature_df["name"] == name].iloc[0]
print(t)
print(encode_row(t))
print(decode(encode_row(t)))

creature_df["encoded"] = creature_df.apply(encode_row, axis=1)
creature_df["encoded_length"] = creature_df["encoded"].apply(len)

# creature_df = creature_df[creature_df["encoded_length"] < 20]

total_words = max(creature_df["encoded_length"])
largest_encode = max(creature_df["encoded_length"])
creature_df[creature_df["encoded_length"] == largest_encode]

input_sequences = []
for seq in creature_df["encoded"]:
#     input_sequences.append(seq)
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

print(len(label[0]))
print(total_words)

import os
# Include the epoch in the file name (uses `str.format`)

#model.load_weights("training_5/cp-0001.ckpt")
checkpoint_path = "training_5/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=1963)

history = model.fit(predictors, label, epochs=300, verbose=1, callbacks=[cp_callback])
 

model.load_weights("training_5/cp-0007.ckpt")

len(input_sequences)

thingy = [tokens["{2}"], tokens["{r}"], tokens["{r}"], tokens["{r}"]]
for i in range(20):
    X = pad_sequences([thingy], maxlen=max_sequence_len - 1, padding='pre')
    thingy.append(np.argmax(model.predict(X)))

print(thingy)
print(decode(thingy))
