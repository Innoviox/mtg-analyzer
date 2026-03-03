import marimo

__generated_with = "0.14.17"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import json
    import requests
    import tabulate as tb
    import numpy as np
    import os
    import tqdm
    import time
    from urllib.parse import quote
    return json, pd, quote, requests, time, tqdm


@app.cell
def _(json, pd, requests):
    URL = "https://data.scryfall.io/default-cards/default-cards-20260303100731.json"
    full_scryfall_df = pd.DataFrame(json.loads(requests.get(URL).text))
    full_scryfall_df[full_scryfall_df['name'] == 'Lightning Bolt']
    return (full_scryfall_df,)


@app.cell
def _(full_scryfall_df):
    set(list(full_scryfall_df['frame'].values))
    return


@app.cell
def _(full_scryfall_df):
    df = full_scryfall_df[['name',                       # the name of the card - not technically necessary but helpful for debugging
                           'mana_cost',                  # what type of mana the card costs to summon
                           'cmc',                        # how much mana the card costs
                           'type_line',                  # the type of the card (creature, sorcery, etc)
                           'oracle_text',                # what the card does
                           'power', 'toughness',         # the strength of the card if it's a creature
                           'colors', 'color_identity',   # more info on what type of mana the card costs
                           'keywords',                   # the keywords on the card (more on this later)
                           'set', 'set_type', 'released_at',         # when the card was released
                           'rarity',                     # how much the card was printed
                           'games',                      # games tells if it is legal online or in paper (we exclude online-only cards)
                           'legalities', 'card_faces', 'frame']]                # which formats the card is legal in

    df = df[df['games'].apply(lambda i: 'paper' in i)]
    df = df.sort_values(by=['released_at'])
    return (df,)


@app.cell
def _(df):
    def get_all_sets(card):
        return df[df['name'] == card]['set'].values
    return


@app.cell
def _(quote, requests, time):
    def get_price_history(name, set):
        time.sleep(1) # ratelimit
        r = requests.get(PRICE_URL.format(id=quote(f"{name} [{set.upper()}]")))
        try:
            q = r.text.split("d += ")
            q = q[1:-1] + [q[-1].split(";")[0]]
            q = [i.strip('"; \n\\n').split(", ") for i in q]
            q = {a: float(b) for a, b in q}
            return q
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            print("price error", name, set)
            print(r.text)

    PRICE_URL = "https://www.mtggoldfish.com/price_history_component?card_id={id}&type=paper&price_type=card"

    # get_price_history("Lightning Bolt", "m10")
    return (get_price_history,)


@app.function
def get_cheapest(card):
    ...


@app.cell
def _(full_scryfall_df):
    full_scryfall_df[full_scryfall_df['name'].str.contains('Hanweir Militia Captain')]
    return


@app.cell
def _(df):
    set_type_order = ['expansion', 'core']
    frame_type_order = ['2003', '2015', 'future', '1997', '1993']

    def get_oset(card):
        rows = df[df['name'].str.contains(card)].sort_values(by=['frame', 'released_at'], key=lambda col: col.apply(frame_type_order.index) if col.name == 'frame' else col)
        # rows = rows.sort_values(by='released_at')
        for st in set_type_order:
            r = rows[rows['set_type'] == st]
            if len(r) > 0:
                # return r
                return r.iloc[0]['set']
        # print(rows)
        if len(rows) > 0:
            return rows.iloc[0]['set'] # fallback if not in set_type_order
        print("error", card)
        # rows = rows[rows['set_type'] == 'expansion']
    # df[df['name'] == 'True-Name Nemesis'] 
    # {'promo', 'alchemy', 'memorabilia', 'premium_deck', 'from_the_vault', 'commander', 'eternal', 'box', 'masterpiece', 'draft_innovation', 'archenemy', 'masters', 'spellbook', 'minigame', 'vanguard', 'treasure_chest', 'token', 'expansion', 'core', 'arsenal', 'planechase', 'starter', 'funny', 'duel_deck'}



    # rows = df[df['name'] == 'Lightning Bolt']
    # b = rows[rows['set_type'] == 'expansion'].sort_values(by='frame', key=lambda col: col.apply(frame_type_order.index))
    # len(b)

    # print(get_oset('Lightning Bolt'))
    # print(get_oset('Myriad Landscape'))
    # print(get_oset('Spellseeker'))
    # print(get_oset('Grim Lavamancer'))
    # print(get_oset('Tarmogoyf'))
    return (get_oset,)


@app.cell
def _():
    cube = open("cubes/moderncubexiv.txt").read().split("# maybeboard")[0].split("\n")[1:-2]
    histories = {}
    errors = []
    return cube, errors, histories


@app.cell
def _(cube, errors, get_oset, get_price_history, histories, tqdm):

    for name in tqdm.tqdm(cube):
        oset = get_oset(name)
        if oset is not None:
            histories[(name, oset)] = get_price_history(name, oset)  
        else:
            print("error", name)
            errors.append(name)
    return


@app.cell
def _(histories):
    summed_histories = {}
    base = histories[list(histories.keys())[0]]
    for k in base.keys():
        if all(k in i or len(i) == 0 for i in histories.values()):
            summed_histories[k] = sum(i.get(k, 0) for i in histories.values())

    return (summed_histories,)


@app.cell
def _(summed_histories):
    import matplotlib.pyplot as plt

    dates = list(summed_histories.keys())
    values = list(summed_histories.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, values, linewidth=1.4, color="#2563EB")
    ax.tick_params(axis="x", rotation=45)
    step = max(1, len(dates) // 10)
    ax.set_xticks(range(0, len(dates), step))
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title("Value over Time")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("plot_output.png", dpi=150)
    plt.show()

    return


if __name__ == "__main__":
    app.run()
