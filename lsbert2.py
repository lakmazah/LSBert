from transformers import BertTokenizer, BertForMaskedLM
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from scipy.special import softmax
import ranker
import nltk

class LSBertSimplifier:
    def __init__(self, model_name="bert-large-uncased-whole-word-masking", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


    def simplify(self, sentence, target_word, target_index=None, top_k=10):
        words = nltk.word_tokenize(sentence)
        
        if target_index is None:
            try:
                target_index = words.index(target_word)
            except ValueError:
                raise ValueError(f"Target word '{target_word}' not found in sentence.")
        
        # Mask only the target word at the given index
        words_masked = words.copy()
        words_masked[target_index] = self.tokenizer.mask_token
        sentence_masked = ' '.join(words_masked)
        
        # Tokenize and send through model
        encoding = self.tokenizer(sentence_masked, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**encoding)
        logits = outputs.logits

        # Locate mask token position
        mask_token_index = (encoding['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        if len(mask_token_index) == 0:
            raise ValueError("Mask token not found in input.")

        top = logits[0, mask_token_index[0]].topk(top_k)
        tokens = self.tokenizer.convert_ids_to_tokens(top.indices)
        scores = softmax(top.values.cpu().numpy())

        return list(zip(tokens, scores))

    def simplify_full_sentence(self, sentence, alpha = 1.0, beta = 1.0, gamma = 10.0, num_sentences=10, num_word_cands = 10, cwi_func=None):
        words = nltk.word_tokenize(sentence)

        if cwi_func:
            complex_targets = cwi_func(words)
        else:
            # default: treat all words as complex
            complex_targets = [{"word": w, "index": i} for i, w in enumerate(words)]

        simplified_versions = [words.copy() for _ in range(top_k)]
        print(complex_targets)
        for target in complex_targets:
            target_word = target["word"]
            target_index = target["index"]

            raw_candidates = self.simplify(sentence, target_word, target_index, top_k=num_word_cands)
            ranked = ranker.rank_candidates(
                source_word=target_word,
                sentence=sentence,
                candidates=raw_candidates,
                tokenizer=self.tokenizer,
                model=self.model,
                word_freq_dict=self.word_freq_dict,
                fasttext_dico=self.fasttext_dico,
                fasttext_emb=self.fasttext_emb,
                alpha = alpha,
                beta = beta,
                gamma = gamma
            )

            top_candidates = [cand for cand, _ in ranked if cand.lower() != target_word.lower()][:top_k]

            for i in range(min(len(top_candidates), num_sentences)):
                simplified_versions[i][target_index] = top_candidates[i]

        return [' '.join(words) for words in simplified_versions]

    def choose_most_similar(input_sentence, candidate_sentences):
        # Encode input and candidates
        embeddings = self.sbert_model.encode([input_sentence] + candidate_sentences)
        
        input_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        
        #Compute cosine similarities
        similarities = util.cos_sim(input_emb, candidate_embs)[0]
        
        # Index of the most similar candidate
        best_idx = similarities.argmax().item()
        
        return candidate_sentences[best_idx], similarities[best_idx].item()