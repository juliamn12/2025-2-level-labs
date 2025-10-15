"""
Spellcheck starter
"""

# pylint:disable=unused-variable, duplicate-code, too-many-locals
from lab_1_keywords_tfidf.main import (
    clean_and_tokenize,
    remove_stop_words,
)
from lab_2_spellcheck.main import (
    build_vocabulary,
    find_out_of_vocab_words,
    find_correct_word,
)

def main() -> None:
    """
    Launches an implementation.
    """
    with open("assets/Master_and_Margarita_chapter1.txt", "r", encoding="utf-8") as file:
        text = file.read()
    with open("assets/stop_words.txt", "r", encoding="utf-8") as file:
        stop_words = file.read().split("\n")
    with (
        open("assets/incorrect_sentence_1.txt", "r", encoding="utf-8") as f1,
        open("assets/incorrect_sentence_2.txt", "r", encoding="utf-8") as f2,
        open("assets/incorrect_sentence_3.txt", "r", encoding="utf-8") as f3,
        open("assets/incorrect_sentence_4.txt", "r", encoding="utf-8") as f4,
        open("assets/incorrect_sentence_5.txt", "r", encoding="utf-8") as f5,
    ):
        sentences = [f.read() for f in (f1, f2, f3, f4, f5)]
    tokens = clean_and_tokenize(text)
    tokens_without_stopwords = remove_stop_words(tokens, stop_words)
    vocabulary = build_vocabulary(tokens_without_stopwords)
    corrections = []
    for s in sentences:
        words = clean_and_tokenize(s)
        out_of_vocab = find_out_of_vocab_words(words, vocabulary)
        for word in out_of_vocab:
            use_jaccard = find_correct_word(word, vocabulary, "jaccard")
            use_frequency_based = find_correct_word(word, vocabulary, "frequency-based")
            corrections.append({"word": word, "jaccard": use_jaccard, "frequency-based": use_frequency_based})
            print(f"Word: {word}")
            print(f"Jaccard: {use_jaccard}")
            print(f"Frequency-based: {use_frequency_based}")
    result = corrections
    assert result, "Result is None"


if __name__ == "__main__":
    main()
