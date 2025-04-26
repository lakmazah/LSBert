from nltk.tokenize import word_tokenize
import string

def manual_cwi(target_words):
    """Returns a function that flags words in a sentence that match any in the target list."""
    target_words_lower = [t.lower() for t in target_words]

    def cwi_func(words):
        return [
            {"word": w, "index": i}
            for i, w in enumerate(words)
            if w.lower() in target_words_lower
        ]

    return cwi_func

def frequency_based_cwi(words, frequency_dict, threshold=3e3):
    """Returns words with frequency below a threshold."""
    return [
        {"word": w, "index": i}
        for i, w in enumerate(words)
        if frequency_dict.get(w.lower(), 0.0) < threshold
    ]

def bert_cwi_func(words, cwi, threshold=0.8):
    sentence = ' '.join(words)
    complex_words = cwi.predict_complex_words(sentence, threshold=threshold)

    complex_targets = []
    used_indices = set()

    for word, prob in complex_words:
        for idx, w in enumerate(words):
            if idx in used_indices:
                continue  # already matched this index
            if w.lower() == word.lower():
                complex_targets.append({"word": w, "index": idx})
                used_indices.add(idx)
                break  # only match once per complex word

    return complex_targets

class ComplexWordIdentifier:
    def __init__(self, model, freq_dict):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clf = model
        self.freq_dict = freq_dict
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.bert_model.eval()

    def predict_complex_words(self, sentence, threshold=0.9):
        #Tokenize the sentence
        words = word_tokenize(sentence)

        #Tokenize with BERT and get word_ids
        encoding = self.tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True)
        word_ids = encoding.word_ids()
        inputs = {k: v.to(self.device) for k, v in encoding.items() if k != 'offset_mapping'}

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        hidden_states = outputs.last_hidden_state.squeeze(0)

        #Group embeddings by word_id
        word_embeds = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue  #skip special tokens
            if word_id not in word_embeds:
                word_embeds[word_id] = []
            word_embeds[word_id].append(hidden_states[idx])

        #Predict complexity per real word
        complex_words = []

        for word_idx, word in enumerate(words):
            if word_idx not in word_embeds:
                continue  # skip if no embedding (rare)
            
            if all(char in string.punctuation for char in word):
              continue  # punctuation, don't predict

            token_embeddings = word_embeds[word_idx]
            avg_embedding = torch.stack(token_embeddings).mean(dim=0).cpu().numpy()

            #Frequency feature
            freq = self.freq_dict.get(word.lower(), 1)
            freq_log = np.log1p(freq)

            feature_vector = np.concatenate((avg_embedding, [freq_log]))
            feature_vector = feature_vector.reshape(1, -1)

            proba = self.clf.predict_proba(feature_vector)[0][1]

            if proba >= threshold:
                complex_words.append((word, proba))

        return complex_words