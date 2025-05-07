# Lexical Simplification Pipeline

Lexical simplification aims to make text easier to understand by identifying complex words and replacing them with simpler alternatives, while preserving the original meaning.

This project implements a full simplification pipeline including:
- **Complex Word Identification (CWI)** using both a frequency-based and a BERT+logistic regression model.
- **Candidate Generation** using a masked BERT language model.
- **Candidate Ranking** using a combination of sentence pseudo-probability, word frequency, and semantic similarity.
- **Sentence Selection** using Sentence-BERT to select the simplification closest in meaning to the original.
- **CEFR Classification** of the original and simplified sentence using a fine-tuned DeBERTa model.

The system can be used to simplify sentences and evaluate how much simpler (and how faithful) the output is.

---

## Setup Instructions

### Clone the repo and install dependencies
```bash
git clone https://github.com/lakmazah/LSBert.git
cd LSBert
pip install -r requirements.txt
```
## Download Models and Data

## Logistic-BERT CWI Model
Place the model in `./models/`:
- `cwi_logistic_regression.joblib`

### BERT Masked Language Model (for candidate generation)
This system uses [bert-base-uncased](https://huggingface.co/bert-base-uncased) and
['bert-large-uncased-whole-word-masking'](https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking)
from HuggingFace

### Sentence Similarity Model
Used to measure semantic preservation via cosine similarity.
We use [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from Sentence-Transformers.

### CEFR Classifier
Download the DeBERTA model: https://www.kaggle.com/models/vinaxue/cefr-classifier-bert/pyTorch/deberta

### Word Frequency Files
The following should be in `./frequencies/`:
- `freq_dict.json` (combined SUBTLEX_US and wiki frequencies)

### FastText Embeddings
Place the .vec file in ./embeddings/
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
```

### Jupyter Notebooks
Refer to 'ls_interactive.ipynb' for the simplifier model. The core is shown below.

Setup
```
fasttext_words, fasttext_vecs = load_fasttext("./embeddings/crawl-300d-2M.vec")

with open("./frequencies/freq_dict.json", "r") as f:
    freq_dict = json.load(f)

clf = joblib.load("./models/cwi_logistic_regression.joblib")
cwi = ComplexWordIdentifier(clf, freq_dict)

simplifier = LSBertSimplifier()
simplifier.word_freq_dict = freq_dict
simplifier.fasttext_dico = fasttext_words
simplifier.fasttext_emb = fasttext_vecs

num_candidates_per_word = 50
num_sentences = 10
alpha = 1.0 # lm score multiplier
beta = 1.0 # frequency score multiplier
gamma = 30 # similarity score multiplier
```

Simplification with choice of CWI function
```
simplified = simplifier.simplify_full_sentence(
        sentence=sent,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        num_sentences=num_sentences,
        num_word_cands=num_candidates_per_word,
        cwi_func=lambda words: frequency_based_cwi(words, freq_dict, threshold=0.3)
    )
```
OR
```
simplified = simplifier.simplify_full_sentence(
        sentence=sent,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        num_sentences=num_sentences,
        num_word_cands=num_candidates_per_word,
        cwi_func=lambda words: bert_cwi_func(words, cwi, threshold=0.5)
    )
```
Outputting simplifed sentence

```
final = simplifier.choose_most_similar(sentence, simplified)
print(final[0])

#final[1] is the simliarity score
```

Refer to 'cefr-classifier.ipynb' for the CEFR classifier
