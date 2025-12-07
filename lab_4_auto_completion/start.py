"""
Auto-completion start
"""

from lab_3_generate_by_ngrams.main import (
    BeamSearchTextGenerator,
    GreedyTextGenerator,
)

# pylint:disable=unused-variable
from lab_4_auto_completion.main import NGramTrieLanguageModel, PrefixTrie, WordProcessor


def main() -> None:
    """
    Launches an implementation.

    In any case returns, None is returned
    """
    with open("./assets/hp_letters.txt", "r", encoding="utf-8") as letters_file:
        hp_letters = letters_file.read()
    with open("./assets/ussr_letters.txt", "r", encoding="utf-8") as text_file:
        ussr_letters = text_file.read()
    word_processor = WordProcessor(end_of_sentence_token="<EOS>")
    encoded_hp = word_processor.encode_sentences(hp_letters)
    prefix_trie = PrefixTrie()
    prefix_trie.fill(encoded_hp)
    suggestions = prefix_trie.suggest((2,))
    if suggestions:
        first_suggestion = suggestions[0]
        decoded = word_processor.decode(first_suggestion)
        print(f"Prefix Trie suggestion: {decoded.replace('<EOS>', ' ').strip()}")

    n_gram_size = 5
    ngram_model = NGramTrieLanguageModel(encoded_hp, n_gram_size)
    ngram_model.build()
    greedy_generator = GreedyTextGenerator(ngram_model, word_processor)
    beam_generator = BeamSearchTextGenerator(ngram_model, word_processor, beam_width=3)
    prompt = "Dear"
    greedy_before_result = greedy_generator.run(seq_len=30, prompt=prompt)
    beam_before_result = beam_generator.run(prompt=prompt, seq_len=30)
    ussr_encoded = word_processor.encode_sentences(ussr_letters)
    ngram_model.update(ussr_encoded)
    greedy_after_result = greedy_generator.run(seq_len=30, prompt=prompt)
    beam_after_result = beam_generator.run(prompt=prompt, seq_len=30)
    print(f"Before update: Greedy = {greedy_before_result}, Beam = {beam_before_result}")
    print(f"After update: Greedy = {greedy_after_result}, Beam = {beam_after_result}")
    result = decoded
    assert result, "Result is None"


if __name__ == "__main__":
    main()
