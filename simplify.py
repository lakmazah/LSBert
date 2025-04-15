import os
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from scipy.special import softmax

ps = PorterStemmer()

def load_fasttext(path):
    words = []
    vecs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and len(line.split()) < 10:
                continue  # header line
            parts = line.rstrip().split(' ')
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))
    return words, np.stack(vecs)

def getWordCount(path):
    word2count = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word2count[parts[0]] = float(parts[1])
    return word2count

def substitution_generation(source_word, pre_tokens, pre_scores, num_selection=10):
    cur_tokens = []
    source_stem = ps.stem(source_word)
    for token in pre_tokens:
        if token.startswith('##') or token == source_word:
            continue
        if ps.stem(token) == source_stem:
            continue
        cur_tokens.append(token)
        if len(cur_tokens) == num_selection:
            break
    return cur_tokens or pre_tokens[:num_selection]

def cross_entropy_word(X, i, pos):
    X = softmax(X, axis=1)
    return -np.log10(X[i, pos])

def get_score(sentence, tokenizer, model):
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids]).to(model.device)
    sentence_loss = 0
    with torch.no_grad():
        for i in range(1, len(tokens) - 1):
            original = tokens[i]
            tokens[i] = '[MASK]'
            masked_ids = tokenizer.convert_tokens_to_ids(tokens)
            masked_tensor = torch.tensor([masked_ids]).to(model.device)
            outputs = model(masked_tensor)
            logits = outputs.logits
            loss = cross_entropy_word(logits[0].cpu().numpy(), i, input_ids[i])
            sentence_loss += loss
            tokens[i] = original
    return np.exp(sentence_loss / (len(tokens) - 2))

def simplify_word(sentence, target_word, tokenizer, model, fasttext_dico, fasttext_emb, word_count, num_candidates=10):
    tokenized = tokenizer.tokenize(sentence.lower())
    if target_word.lower() not in sentence.lower():
        raise ValueError("Target word not found in sentence.")
    sentence_masked = sentence.lower().replace(target_word.lower(), '[MASK]')
    input_ids = tokenizer.encode(sentence_masked, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    mask_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
    topk = logits[0, mask_index].topk(num_candidates * 2)
    pre_tokens = tokenizer.convert_ids_to_tokens(topk.indices.cpu().numpy())
    pre_scores = topk.values.cpu().numpy()

    candidates = substitution_generation(target_word, pre_tokens, pre_scores, num_selection=num_candidates)
    results = []
    for word in candidates:
        freq = word_count.get(word, 0)
        sim = cosine_similarity(
            fasttext_emb[fasttext_dico.index(target_word)].reshape(1, -1),
            fasttext_emb[fasttext_dico.index(word)].reshape(1, -1)
        )[0][0] if target_word in fasttext_dico and word in fasttext_dico else 0
        lm_score = get_score(sentence.lower().replace(target_word.lower(), word), tokenizer, model)
        results.append((word, lm_score, freq, sim))

    return sorted(results, key=lambda x: x[1])