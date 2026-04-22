import requests
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from scipy.signal import medfilt
import re

API_URL = "https://en.wikipedia.org/w/api.php"

def get_revisions(title, fields="timestamp|user|size|comment", limit=500):
    revisions = []
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "titles": title,
        "rvlimit": limit,
        "rvprop": fields,
        "rvslots": "main",
    }

    while True:
        HEADERS = {
            "User-Agent": "AzuraResearchBot/1.0 (contact: azuranishio@gmail.com)"
        }
        
        r = requests.get(API_URL, params=params, headers=HEADERS)

        response = r.json()
        
        pages = response["query"]["pages"]

        for page_id in pages:
            if "revisions" in pages[page_id]:
                revisions.extend(pages[page_id]["revisions"])

        # pagination
        if "continue" in response:
            params.update(response["continue"])
        else:
            break
    
    
    fields_arr = fields.split("|")
    rows = []
    
    for rev in revisions:
        row = {}

        for field in fields_arr:
            if field == "content":
                # special case: nested structure
                if "slots" in rev and "main" in rev["slots"]:
                    row["content"] = rev["slots"]["main"].get("*")
                else:
                    row["content"] = None
            else:
                row[field] = rev.get(field)

        rows.append(row)
    
    try:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except:
        print(rows[0])
    
    
    # time stuff
    df = pd.DataFrame(rows)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    
    # normalize time
    t0 = df["timestamp"].iloc[0]
    df["t"] = (df["timestamp"] - t0).dt.total_seconds()
    df["t_days"] = df["t"] / 86400
    
    return df

def unpluralize(str):
    unpluralized_name = str
    if unpluralized_name.endswith("s"):
        unpluralized_name = unpluralized_name + "-"
        unpluralized_name = unpluralized_name.replace("s-", "")
    return unpluralized_name

class WikiError(Exception):
    pass

class Article:
    def __init__(self, data: pd.DataFrame, name: str, fields: list):
        self.data = data
        self.name = name
        self.fields = fields
        
    def get_field(self, field:str):
        return self.data[field]

    def __getattr__(self, field: str):
        # allows article.timestamps, article.user, etc.
        unpluralized_name = unpluralize(field)
        
        if unpluralized_name in self.fields:
            return self.get_field(unpluralized_name)
        raise AttributeError(f"Article {self.name} has no attribute '{field}'")

    def resample(self, field: str, days: int, method="last"):
        df = self.data.copy()
        unpluralized_field = unpluralize(field)


        if "timestamp" not in df.columns:
            raise ValueError("No timestamp column available")

        df = df.sort_values("timestamp")
        df = df.set_index("timestamp")

        rule = pd.to_timedelta(days, unit="D")
        
        if method == "count":
            return df.resample(rule).count()
        
        if unpluralized_field == "timestamp":
            if method == "last":
                return df.resample(rule).apply(lambda x: x.index.max())
            elif method == "first":
                return df.resample(rule).apply(lambda x: x.index.min())
            else:
                raise ValueError("Unsupported method for timestamp")

        if method == "last":
            return df[unpluralized_field].resample(rule).last()

        elif method == "first":
            return df[unpluralized_field].resample(rule).first()

        elif method == "mean":
            return df[unpluralized_field].resample(rule).mean()
        
        elif method == "diff_median":
            s = df[unpluralized_field].diff()
            return s.resample(rule).median()



        else:
            raise ValueError(f"Unknown method {method}")



class Wikipedia:
    def __init__(self):
        self.articles = {}
        self.load_cache()
        
    def save_cache(self, path="cache"):
        os.makedirs(path, exist_ok=True)
    
        for title, article in self.articles.items():
            safe_title = title.replace("/", "_")
    
            data_path = f"{path}/{safe_title}.json"
            meta_path = f"{path}/{safe_title}_meta.json"
    
            # save dataframe
            article.data.to_json(data_path, orient="records", date_format="iso")
    
            # save metadata
            with open(meta_path, "w") as f:
                json.dump({
                    "name": article.name,
                    "fields": article.fields
                }, f)
                
    def load_cache(self, path="cache"):
        
        if not os.path.exists(path):
            return 

        for file in os.listdir(path):
            if file.endswith("_meta.json"):
                meta_path = os.path.join(path, file)
                data_path = meta_path.replace("_meta.json", ".json")

                with open(meta_path) as f:
                    meta = json.load(f)

                df = pd.read_json(data_path)

                article = Article(df, meta["name"], meta["fields"])
                self.articles[meta["name"]] = article            

    def get_article(self, title: str, fields: str):
        fields_arr = [f.strip() for f in fields.split(",") if f.strip()]

        # check if article is cached
        if title in self.articles:
            article = self.articles[title]

            # if it is, fill missing fields
            missing_fields = [f for f in fields_arr if f not in article.fields]

            if missing_fields:
                new_rows = get_revisions(title, "|".join(missing_fields))
                new_df = pd.DataFrame(new_rows)

                article.data = article.data.join(new_df, how="outer")
                
                article.fields.extend(missing_fields)
                
            self.save_cache()
                
            return article

        # if not cached fetch
        else:
            query_fields = "|".join(fields_arr)
            rows = get_revisions(title, query_fields)
            df = pd.DataFrame(rows)

            article = Article(df, title, fields_arr)
            self.articles[title] = article

            self.save_cache()
            
            return article
    



def duplicate_last(arr: pd.Series):
    return pd.concat([arr, arr.iloc[-1:]])