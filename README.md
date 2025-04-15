# LSBert

A minimal and modern implementation of LSBert1.0 for lexical simplification.

## Overview

Given a sentence and a target word, `simplify_word()` suggests simpler alternatives ranked by:
- BERT-based language model score
- Word frequency
- Semantic similarity using FastText

## Dependencies

pip install -r requirements.txt

This notebook expects the FastText .vec and word frequency .txt files to be located in your Google Drive at:

/MyDrive/LSBERT/embeddings/crawl-300d-2M.vec  
/MyDrive/LSBERT/frequency_merge_wiki_child.txt

## Example Usage

```python
from simplify import simplify_word, load_fasttext, getWordCount
from transformers import BertTokenizer, BertForMaskedLM

# Load models
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking")
model = BertForMaskedLM.from_pretrained("bert-large-uncased-whole-word-masking").to("cpu")

# Load resources
fasttext_words, fasttext_vecs = load_fasttext("path/to/crawl-300d-2M.vec")
word_freqs = getWordCount("path/to/frequency_merge_wiki_child.txt")

# Run simplification
results = simplify_word(
    sentence="The lecture was very tedious.",
    target_word="tedious",
    tokenizer=tokenizer,
    model=model,
    fasttext_dico=fasttext_words,
    fasttext_emb=fasttext_vecs,
    word_count=word_freqs
)

for word, lm_score, freq, sim in results:
    print(f"{word:<15} | LM Score: {lm_score:.4f} | Freq: {freq:.0f} | Similarity: {sim:.2f}")
