import pandas as pd
import numpy as np

def load_subtlex(path):
    df = pd.read_excel(path)
    return dict(zip(df["Word"].str.lower(), df["FREQcount"]))

def load_wiki_child(path):
    freq_dict = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word, freq = parts
                freq_dict[word.lower()] = float(freq)
    return freq_dict

def merge_frequencies(dict1, dict2):
    merged = dict1.copy()
    for word, freq in dict2.items():
        if word in merged:
            merged[word] = max(merged[word], freq)
        else:
            merged[word] = freq
    return merged

def load_fasttext(path):
    words = []
    vecs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) < 10:
                continue
            parts = line.rstrip().split(' ')
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))
    return words, np.stack(vecs)