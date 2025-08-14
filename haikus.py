import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import json
    import requests
    import matplotlib.pyplot as plt
    import re
    import numpy as np
    import datetime as dt
    from sklearn.linear_model import LinearRegression
    import statsmodels.api as sm 
    import statistics
    from hyphen import Hyphenator
    import cmudict
    from functools import cache
    return cache, cmudict, json, pd, requests


@app.cell
def _(json, pd, requests):
    URL = "https://data.scryfall.io/default-cards/default-cards-20250813211854.json"
    full_scryfall_df = pd.DataFrame(json.loads(requests.get(URL).text))
    full_scryfall_df.head()
    return (full_scryfall_df,)


@app.cell
def _(full_scryfall_df, json, requests):
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
    return (df,)


@app.cell
def _(df):
    df['oracle_text'] = df['oracle_text'].str.replace(r'\(.*\)', '', regex=True)
    df['num_lines'] = df['oracle_text'].map(lambda i: len(i.split("\n")) if isinstance(i, str) else 0)
    haiku_df = df[df['num_lines'] == 3]
    return (haiku_df,)


@app.cell
def _(haiku_df):
    def smooth(text):
        for k, v in {
            '+': 'plus',
            '-': 'minus',
            '{T}': 'tap',
            '{W}': 'white',
            '{U}': 'blue',
            '{B}': 'black',
            '{R}': 'red',
            '{G}': 'green',
            '{C}': 'colorless',
            '{3}': 'three',
            '{X}': 'x',
            '{0}': 'zero',
            '{2}': 'two', 
            '{4}': 'four', 
            '{9}': 'nine', 
            '{1}': 'one', 
            '{8}': 'eight', 
            '{6}': 'six', 
            '{10}': 'ten', 
            '{5}': 'five', 
            '{S}': 'snow',
            '{B/R}': 'black or red', # rakdos?
            '{B/G}': 'black or green',
            '{W/B}': 'white or black',
            '{R/W}': 'red or white', 
            '{U/R}': 'blue or red', 
            '{G/W}': 'green or white',
            '{20}': 'twenty',
            '{B/P}': 'black phyrexian', 
            '{R/P}': 'red phyrexian', 
            '{7}': 'seven', 
            '{R/G}': 'red or green', 
            '{G/U}': 'green or blue', 
            '{E}': 'energy',
            '{U/B}': 'blue or black', 
            '{W/U}': 'white or blue', 
            '{W/P}': 'white phyrexian',
            '{U/P}': 'blue phyrexian', 
            '{G/P}': 'green phyrexian', 

        }.items():
            if k in text:
                text = text.replace(k, ' ' + v + ' ')

        if '{' in text:
            print(text)
            input()

        return text

    haiku_df['smooth'] = haiku_df.oracle_text.apply(smooth)
    return


@app.cell
def _():
    manual = {a: int(b) for a, b in [i.split() for i in open('manual.txt').read().split("\n") if len(i.split()) == 2]}
    import ast
    words = ast.literal_eval(open('words.txt').read())
    return manual, words


@app.cell
def _(cache, cmudict, haiku_df, manual, words):
    # from https://stackoverflow.com/questions/49581705/using-cmudict-to-count-syllables
    from tqdm.auto import tqdm
    tqdm.pandas()

    unknown = []
    @cache
    def lookup_word(word_s):
        return cmudict.dict().get(word_s)

    @cache
    def count_syllables(word_s):
        word_s = word_s.lower().strip('-.,:—;/"')
        if word_s in manual:
            return manual[word_s]
        if word_s in words:
            return words[word_s]
        
        count = 0
        phones = lookup_word(word_s) # this returns a list of matching phonetic rep's
    
        if phones:                   # if the list isn't empty (the word was found)
            phones0 = phones[0]      #     process the first
            count = len([p for p in phones0 if p[-1].isdigit()]) # count the vowels
        else:
            # print(word_s)
            unknown.append(word_s)
        
        words[word_s] = count
        return count

    def sentence_syllables(sent):
        count = 0
        for word in sent.split():
            count += count_syllables(word)
        return count

    for i in range(3):
        haiku_df[f'count_{i}'] = haiku_df.smooth.progress_apply(lambda s: sentence_syllables(s.split("\n")[i]))
    return


@app.cell
def _(haiku_df):
    haiku_df
    return


@app.cell
def _(haiku_df):
    HAIKU = [5, 7, 5]
    def score(row):
        s = 0
        for i in range(3):
            s += abs(row[f'count_{i}'] - HAIKU[i])
        return s

    haiku_df['score'] = haiku_df.apply(score, axis=1)
    return


@app.cell
def _(haiku_df):
    haiku_df.sort_values(by=['score'])
    return


if __name__ == "__main__":
    app.run()
