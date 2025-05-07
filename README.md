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
['bert-large-uncased-whole-word-masking'] (https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking)
from HuggingFace

### Sentence Similarity Model
Used to measure semantic preservation via cosine similarity.
We use [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) from Sentence-Transformers.

### CEFR Classifier
Download the DeBERTA model: https://www.kaggle.com/models/vinaxue/cefr-classifier-bert/pyTorch/deberta

### Word Frequency Files
The following should be in `./frequencies/`:
- `freq_dict.json` (combined SUBTLEX_US and wiki frequencies)
- `wiki_freq.txt` (Wikipedia-based word counts)
- `simplewiki_freq.csv` (SimpleWiki word counts)

### FastText Embeddings
Place the .vec file in ./embeddings/
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
```

### Jupyter Notebooks
Refer to 'ls_interactive.ipynb' for the simplifier model
Refer to 'cefr-classifier.ipynb' for the CEFR classifier
