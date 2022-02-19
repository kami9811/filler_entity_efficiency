import typing
from typing import Any, Dict, List, Text

from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.components import Component
import MeCab
from rasa.shared.nlu.training_data.message import Message

class MecabTokenizer(Tokenizer, Component):

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        # mt = MeCab.Tagger()
        # parsed = mt.parse(text)
        mecab = MeCab.Tagger("-Owakati")
        # print(text)
        # print(message)
        # print(attribute)
        parsed = mecab.parse(text)
        # print(parsed)
        # parsed returns token => POS separated by tab in multiple lines
        words = parsed.replace('\n', ' ').split(' ')  # 形式はMeCabによる

        # running_offset=0
        # tokens = []
        # for word in words:
        #     word_offset = text.index(word, running_offset)
        #     word_len = len(word)
        #     running_offset = word_offset + word_len
        #     tokens.append(Token(word, word_offset))
        tokens = self._convert_words_to_tokens(words, text)
        # print(text)
        # print(words)
        # print(tokens)

        return self._apply_token_pattern(tokens)