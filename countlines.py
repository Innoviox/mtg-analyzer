import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import re
import numpy as np
import datetime as dt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
URL = "https://data.scryfall.io/default-cards/default-cards-20221212220657.json"
full_scryfall_df = pd.DataFrame(json.loads(requests.get(URL).text))
full_scryfall_df.head()
df = full_scryfall_df              # which formats the card is legal in
df = df[df['games'].apply(lambda i: 'paper' in i)]
df = df.sort_values(by=['released_at', 'name'])
df = df.drop_duplicates(subset=['name'])
def legal(legalities):
    v = legalities.values()
    if len(set(v)) == 1 and "not_legal" in v:
        return False
    return True
df = df[df['legalities'].apply(legal)]
df = df[~df["type_line"].str.contains("Token", na=False)] # remove tokens
unsets = ['unglued', 'unhinged', 'unstable', 'unsanctioned', 'unfinity']
sets = json.loads(requests.get("https://api.scryfall.com/sets").text)
for s in sets["data"]:
    if s['name'].lower() in unsets:
        df = df[~df["set"].str.contains(s['code'])]
df = df.dropna() # drop NAN values
alpha_release_date = dt.datetime(1993, 8, 5)
df['released_at'] = df['released_at'].apply(lambda i: (dt.datetime.strptime(i, '%Y-%m-%d').year - alpha_release_date.year))
df["num_colored_pips"] = df["mana_cost"].apply(lambda mana_cost: len(re.findall("\{[^\d]\}", str(mana_cost))))
df["num_colors"] = df["colors"].apply(len)
df = df[df["type_line"].str.contains("Creature", na=False)]
def make_int(i):
    try:
        return int(i)
    except ValueError:
        return np.nan
for i in ['cmc', 'power', 'toughness']:
    df[i] = df[i].apply(make_int).astype('Int64')#
df = df.dropna()
df[df["name"].str.contains("Grizzly Bears")]
df[df["name"].str.contains("Coral Eel")]
vanilla_df = df[df["oracle_text"] == ""]
vanilla_df.head()
len(vanilla_df.index) / len(df.index)
mp, mt = max(vanilla_df['power']) + 1, max(vanilla_df['toughness']) + 1
data = np.zeros((mp, mt))
fig, ax = plt.subplots()
for power in range(mp):
    for toughness in range(mt):
        count = len(vanilla_df[(vanilla_df['power'] == power) & (vanilla_df['toughness'] == toughness)]['cmc'].values)
        data[mp - power - 1][toughness] = count
        
        text = ax.text(toughness, mp - power - 1, count,
                       ha="center", va="center", color="w")
ax.set_yticks(np.arange(mp), labels=list(range(mp))[::-1]) # invert power so that 0/0 is the bottom left corner
im = ax.imshow(data)
fig.tight_layout()
plt.show()
data_df = vanilla_df.drop(columns=["mana_cost", "type_line", "oracle_text", "color_identity", "keywords", "set", "games", "legalities"])
data_df.head()
def dummy_list(data_df, one_hot_df, column, predicate=lambda i, j: i == j):
    x = set(data_df.explode(column)[column].values) # get all values from the column
    if np.nan in x: # remove NaNs that might be in there
        x.remove(np.nan)
    
    for i in x: # make the new one-hot column
        one_hot_df[f'{column}_{i}'] = data_df[column].apply(lambda j: int(predicate(i, j)))
one_hot_df = data_df.drop(columns=['colors', 'rarity'])
dummy_list(data_df, one_hot_df, 'colors', predicate=lambda i, j: i in j)
dummy_list(data_df, one_hot_df, 'rarity')
one_hot_df.head()
X = one_hot_df.drop(columns=['name', 'cmc'])
y = one_hot_df['cmc']
X = sm.add_constant(X)
model = sm.OLS(np.asarray(y, dtype=int), 
               np.asarray(X, dtype=int)).fit()
plt.errorbar(model.params, X.columns, xerr=model.bse, fmt='o', ecolor='black', capsize=5)
plt.xlabel("Coefficient")
plt.ylabel("Variable")
plt.title("Linear Regression for Converted Mana Cost of Vanilla Creatures")
plt.show()
def expected_mana_cost(row):
    return sum(row * model.params[1:]) + model.params[0]

one_hot_df['expected_mana_value'] = one_hot_df.drop(columns=['name', 'cmc']).apply(expected_mana_cost, axis=1)
one_hot_df
keywords = set()
for i, row in df.iterrows():
    if row['set'] not in ['afr', '40k', 'clb', 'sld']:
        keywords = keywords.union(row['keywords'])
    else:
        keywords = keywords.union([kw for kw in row['keywords'] if ' ' not in kw]) 
keyword_soup = ' '.join(keywords).lower() # easy way to convert everything to lowercase
removes = [r'(\(.*?\))']
def extract_ability_text(row):
    text = row['oracle_text']
    if text is np.nan or text == '':
        return ''
    for r in removes:
        if m := re.search(r, text):
            start, end = m.span(1) # remove first capturing group
            text = text[:start] + text[end:]
    text = text.replace(',', '').replace(';', '')
    text = text.lower()
    text = text.strip()
    return text
df['ability_text'] = df.apply(extract_ability_text, axis=1)
def is_french_vanilla(row):
    text = row['ability_text']
    for i in text.split():
        if i not in keyword_soup: # check that every word is a kewyord
            return False
    return True
df['is_french_vanilla'] = df.apply(is_french_vanilla, axis=1)
french_vanilla_df = df[df['is_french_vanilla']]
french_vanilla_df.head()
french_vanilla_df[french_vanilla_df['rarity'] == 'mythic']
fv_data_df = french_vanilla_df.drop(columns=['mana_cost', 'type_line', 'color_identity', 'set', 'is_french_vanilla', 'games', 'legalities'])
fv_one_hot_df = fv_data_df.drop(columns=['colors', 'oracle_text', 'keywords', 'rarity'])
dummy_list(fv_data_df, fv_one_hot_df, 'colors', predicate=lambda i, j: i in j)
dummy_list(fv_data_df, fv_one_hot_df, 'rarity')
fv_one_hot_df.head()
keyword_counts = {}
for keyword in keywords:
    counts = fv_data_df.apply(lambda row: row['ability_text'].count(keyword.lower()), axis=1)
    if sum(counts) > 0: # some keywords never appear on french vanilla creatures
        keyword_counts[keyword] = sum(counts)
        fv_one_hot_df[f'keywords_{keyword}'] = counts
X = fv_one_hot_df.drop(columns=['name', 'cmc', 'ability_text'])
X = sm.add_constant(X)
y = fv_one_hot_df['cmc']
model = sm.OLS(np.asarray(y, dtype=int), 
               np.asarray(X, dtype=int)).fit()
plt.errorbar(model.params, X.columns, xerr=model.bse, fmt='o', ecolor='black', capsize=5)
plt.xlabel("Coefficient")
plt.ylabel("Variable")
plt.title("Linear Regression for Converted Mana Cost of French Vanilla Creatures")
plt.show()
