import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def rank_candidates(
    source_word,
    sentence,
    candidates,
    tokenizer,
    model,
    word_freq_dict,
    fasttext_dico,
    fasttext_emb,
    alpha=1.0,
    beta=1.0,
    gamma=5.0,
):
    def get_lm_score(sent):
        tokens = tokenizer(sent, return_tensors="pt").to(model.device)
        with torch.no_grad():
            loss = model(**tokens, labels=tokens["input_ids"]).loss
        return torch.exp(loss).item()  # Perplexity

    def get_freq(word):
        return word_freq_dict.get(word.lower(), 1e-6)

    def get_sim(word1, word2):
        if word1 in fasttext_dico and word2 in fasttext_dico:
            emb1 = fasttext_emb[fasttext_dico.index(word1)].reshape(1, -1)
            emb2 = fasttext_emb[fasttext_dico.index(word2)].reshape(1, -1)
            return cosine_similarity(emb1, emb2)[0][0]
        return 0.0

    orig_score = get_lm_score(sentence)
    ranked = []

    for cand, _ in candidates:
        new_sent = sentence.replace(source_word, cand)
        lm_score = get_lm_score(new_sent)
        freq_score = np.log(get_freq(cand))  # log-freq
        sim_score = get_sim(source_word, cand)

        score = (
            alpha * (-np.log(lm_score)) +
            beta * freq_score +
            gamma * sim_score
        )
        ranked.append((cand, score))

    ranked.sort(key=lambda x: -x[1])
    return ranked