import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    # import pandas as pd
    import json
    import requests
    import os
    import time
    import matplotlib.pyplot as plt
    import tabulate as tb
    return json, os, plt, requests, tb, time


@app.cell
def _(json, os, pd, requests):
    CACHE_PATH = os.path.expanduser("~/Downloads/default-cards-cached.json")

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            raw = json.load(f)
    else:
        meta = requests.get("https://api.scryfall.com/bulk-data/default-cards").json()
        url = meta["download_uri"]
        print(f"Downloading bulk data from {url} ...")
        raw = requests.get(url).json()
        with open(CACHE_PATH, "w") as f:
            json.dump(raw, f)
        print("Saved to cache.")

    full_scryfall_df = pd.DataFrame(raw)
    full_scryfall_df
    return (full_scryfall_df,)


@app.cell
def _(requests):
    sets_raw = requests.get("https://api.scryfall.com/sets").json()

    MRD_DATE = "2003-10-02"
    WAR_DATE = "2019-05-03"
    EXCLUDE_CODES = {"tsb"}  # Time Spiral Timeshifted — rarity=special only, not a standalone booster product

    target_sets = [
        s for s in sets_raw["data"]
        if s["set_type"] in ("expansion", "core")
        and MRD_DATE <= s.get("released_at", "") <= WAR_DATE
        and s["code"] not in EXCLUDE_CODES
    ]
    target_sets.sort(key=lambda s: s["released_at"])
    target_codes = {s["code"] for s in target_sets}

    print(f"Target sets: {len(target_sets)}")
    for s in target_sets:
        print(f"  {s['released_at']}  {s['code'].upper():5}  {s['name']}")
    return MRD_DATE, WAR_DATE, target_codes, target_sets


@app.cell
def _(full_scryfall_df, target_codes):
    ev_cards = full_scryfall_df[
        full_scryfall_df["set"].isin(target_codes) &
        full_scryfall_df["games"].apply(lambda g: "paper" in g)
    ][["name", "set", "rarity", "type_line", "prices"]].copy()

    ev_cards
    return (ev_cards,)


@app.cell
def _(ev_cards):
    def has_mythics(set_meta):
        # Mythic rarity introduced Oct 2008 (Shards of Alara) for expansions,
        # July 2009 (Magic 2010) for core sets.
        if set_meta["set_type"] == "expansion":
            return set_meta["released_at"] >= "2008-10-03"
        elif set_meta["set_type"] == "core":
            return set_meta["released_at"] >= "2009-07-17"
        return False

    def avg_usd(card_list):
        prices = [
            float(c["prices"]["usd"])
            for c in card_list
            if c.get("prices") and c["prices"].get("usd") is not None
        ]
        return sum(prices) / len(prices) if prices else 0.0

    def compute_ev(set_code, set_meta):
        cards = ev_cards[ev_cards["set"] == set_code].to_dict("records")

        # Deduplicate by name, preferring cards that have a USD price.
        # Needed for sets like 9ED that have multiple printings of the same card.
        seen = {}
        for c in sorted(cards, key=lambda x: x.get("prices", {}) is None or x["prices"].get("usd") is None):
            if c["name"] not in seen:
                seen[c["name"]] = c
        cards = list(seen.values())

        mythics   = [c for c in cards if c["rarity"] == "mythic"]
        rares     = [c for c in cards if c["rarity"] == "rare"]
        uncommons = [c for c in cards if c["rarity"] == "uncommon"]
        commons   = [
            c for c in cards
            if c["rarity"] == "common"
            and "Basic Land" not in (c.get("type_line") or "")
        ]

        avg_m = avg_usd(mythics)
        avg_r = avg_usd(rares)
        avg_u = avg_usd(uncommons)
        avg_c = avg_usd(commons)

        if has_mythics(set_meta) and mythics:
            rare_ev = avg_r * (7 / 8) + avg_m * (1 / 8)
        else:
            rare_ev = avg_r

        return {
            "set":         set_code.upper(),
            "set_name":    set_meta["name"],
            "released_at": set_meta["released_at"],
            "total_ev":    round(rare_ev + avg_u * 3 + avg_c * 10, 2),
            "rare_ev":     round(rare_ev, 2),
            "uncommon_ev": round(avg_u * 3, 2),
            "common_ev":   round(avg_c * 10, 2),
            "avg_rare":    round(avg_r, 2),
            "avg_mythic":  round(avg_m, 2),
            "n_rares":     len(rares),
            "n_mythics":   len(mythics),
        }

    return (compute_ev,)


@app.cell
def _(compute_ev, pd, target_sets):
    rows = [compute_ev(s["code"], s) for s in target_sets]
    ev_df = pd.DataFrame(rows).sort_values("released_at").reset_index(drop=True)
    ev_df
    return (ev_df,)


@app.cell
def _(MRD_DATE, WAR_DATE, json, os, pd, requests, time):
    PACK_PRICE_CACHE = "pack_prices_cache.json"

    def load_price_cache():
        if os.path.exists(PACK_PRICE_CACHE):
            with open(PACK_PRICE_CACHE) as f:
                return json.load(f)
        return {}

    def save_price_cache(cache):
        with open(PACK_PRICE_CACHE, "w") as f:
            json.dump(cache, f, indent=2)

    def get_pack_price(group_id):
        """Fetch the market price of a single booster pack for a TCGCSV group."""
        products_resp = requests.get(f"https://tcgcsv.com/tcgplayer/1/{group_id}/products").json()
        results = products_resp.get("results", [])

        # Find the booster pack product: name contains "Booster Pack",
        # and no "Rarity" field in extendedData (which would indicate a single card)
        pack = next(
            (p for p in results
             if "Booster Pack" in p.get("name", "") and not "Booster Packs" in p.get("name", "")
             and not any(e.get("name") == "Rarity" for e in p.get("extendedData", []))),
            None
        )
        if pack is None:
            return None

        prices_resp = requests.get(f"https://tcgcsv.com/tcgplayer/1/{group_id}/prices").json()
        price_entry = next(
            (p for p in prices_resp.get("results", [])
             if p["productId"] == pack["productId"]
             and p.get("subTypeName") == "Normal"),
            None
        )
        return price_entry["marketPrice"] if price_entry else None

    # Fetch all MTG groups from TCGCSV and filter to MRD–WAR date range
    groups_resp = requests.get("https://tcgcsv.com/tcgplayer/1/groups").json()
    tcgcsv_groups = [
        g for g in groups_resp.get("results", [])
        if g.get("publishedOn") and MRD_DATE <= g["publishedOn"][:10] <= WAR_DATE
    ]
    print(f"TCGCSV groups in range: {len(tcgcsv_groups)}")

    # Fetch pack prices with cache + keyboard interrupt recovery
    price_cache = load_price_cache()
    try:
        for g in tcgcsv_groups:
            if g['abbreviation'] == 'RTR':
                print(get_pack_price(g["groupId"]))
            gid = str(g["groupId"])
            if gid in price_cache:
                continue
            price = get_pack_price(g["groupId"])
            price_cache[gid] = {
                "set_name":   g["name"],
                "released_at": g.get("publishedOn", "")[:10],
                "pack_price": price,
            }
            save_price_cache(price_cache)
            time.sleep(0.3)
            print(f"  {g['name']:40} ${price}")
    except KeyboardInterrupt:
        print("Interrupted — progress saved to cache.")

    pack_prices_df = pd.DataFrame(price_cache.values())
    pack_prices_df = pack_prices_df[pack_prices_df["pack_price"].notna()]
    pack_prices_df
    return (pack_prices_df,)


@app.cell
def _(ev_df, pack_prices_df):
    combined_df = ev_df.merge(
        pack_prices_df[["set_name", "pack_price"]],
        on="set_name",
        how="left"
    )
    combined_df["ev_vs_price"] = (combined_df["total_ev"] - combined_df["pack_price"]).round(2)
    combined_df
    return (combined_df,)


@app.cell
def _(combined_df, plt):
    labels      = combined_df["set"].tolist()
    common_evs  = combined_df["common_ev"].tolist()
    uncommon_evs= combined_df["uncommon_ev"].tolist()
    rare_evs    = combined_df["rare_ev"].tolist()
    pack_prices = combined_df["pack_price"].tolist()

    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(18, 6))

    # Stacked EV bars
    ax.bar(x, common_evs,    label="Commons (10)",     color="#93C5FD", width=0.7)
    ax.bar(x, uncommon_evs,  bottom=common_evs,        label="Uncommons (3)",    color="#3B82F6", width=0.7)
    bottoms = [c + u for c, u in zip(common_evs, uncommon_evs)]
    ax.bar(x, rare_evs,      bottom=bottoms,            label="Rare/Mythic slot", color="#1D4ED8", width=0.7)

    # Pack market price overlay
    ax.plot(x, pack_prices, color="#EF4444", linestyle="--", linewidth=1.5,
            marker="o", markersize=3, label="Pack market price (TCGPlayer)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("USD")
    ax.set_title("Booster Pack EV vs. Market Price by MTG Set (MRD → WAR, current prices)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("pack_ev_output.png", dpi=150)
    plt.show()

    return


@app.cell
def _(combined_df, tb):
    cols = ["set", "set_name", "released_at", "total_ev", "pack_price", "ev_vs_price"]
    top5    = combined_df.nlargest(5,  "total_ev")[cols]
    bottom5 = combined_df.nsmallest(5, "ev_vs_price")[cols]
    positive_ev = combined_df[combined_df["ev_vs_price"] > 0][cols].sort_values("ev_vs_price", ascending=False)

    print("=== Top 5 Sets by Pack EV ===")
    print(tb.tabulate(top5.values,    headers=cols, tablefmt="outline", floatfmt=".2f"))
    print()
    print("=== Bottom 5 Sets by Pack EV ===")
    print(tb.tabulate(bottom5.values, headers=cols, tablefmt="outline", floatfmt=".2f"))
    print()
    print("=== Sets Where EV > Pack Price (positive expected value to open) ===")
    if len(positive_ev):
        print(tb.tabulate(positive_ev.values, headers=cols, tablefmt="outline", floatfmt=".2f"))
    else:
        print("  (none — all packs are currently negative EV to open)")
    return


if __name__ == "__main__":
    app.run()
