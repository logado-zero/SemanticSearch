import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def preprocessing(text: str) -> str:
    pre_list = [token.lemma_.lower() 
                        for token in nlp(text)
                        if (token.is_alpha or token.is_digit) and not token.is_stop]
    return " ".join(pre_list)
