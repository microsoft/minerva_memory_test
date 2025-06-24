import os
import random
import string

import tiktoken

import logging

logging.basicConfig(level=logging.INFO)


def get_word_list(filename="words_alpha.txt"):
    """Load a list of words from a file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)
    
    with open(filepath, "r") as f:
        words = f.read().splitlines()
    return words


class ContextGenerator:
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    WORDS = get_word_list()
    
    def __init__(self):
        self.max_length = 4096
        self.num_samples = 10

    @classmethod
    def get_context_length(cls, context):
        return len(cls.tokenizer.encode(context))

    @classmethod
    def encode_and_trim(cls, context, context_length):
        tokens = cls.tokenizer.encode(context)
        if len(tokens) > context_length:
            tokens = tokens[:context_length]

        return cls.tokenizer.decode(tokens)
    
    @classmethod
    def trim_context(cls, context, max_length):
        tokens = cls.tokenizer.encode(context)

        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        trimmed_context = cls.tokenizer.decode(tokens)

        if context[len(trimmed_context)] in [",", " "]:
            if context[len(trimmed_context) - 1] != ":":
                return trimmed_context.rstrip(", ")
        # discard the last token if it is not complete
        trimmed_context = ", ".join(trimmed_context.split(", ")[:-1])
        return trimmed_context

    def generate_context(self, context_type, length=None, num_samples=None):
        valid_context_types = [
            "random_numbers",
            "unique_words",
            "word_pairs",
            "gibberish",
        ]

        if context_type not in valid_context_types:
            raise ValueError(f"Context type {context_type} is invalid")

        if not length:
            length = self.max_length
        if not num_samples:
            num_samples = self.num_samples

        data = []
        for _ in range(num_samples):
            if context_type == "unique_words":
                context_data = self.generate_unique_words(length)
            elif context_type == "random_numbers":
                context_data = self.generate_random_numbers(length)
            elif context_type == "word_pairs":
                context_data = self.generate_word_pairs(length)
            elif context_type == "gibberish":
                context_data = self.generate_gibberish_words(length)

            data.append(context_data)

        logging.info(f"Generated {num_samples} context data of type {context_type}")

        return data
    
    def generate_unique_words(self, length):
        candidate_words = random.sample(self.WORDS, length)
        return self.trim_context(", ".join(candidate_words), length)

    def generate_random_numbers(self, length):
        numbers = [str(random.randint(0, 1000)) for _ in range(length)]
        context = ", ".join(numbers)
        return self.trim_context(context, length)

    def generate_word_pairs(self, length):
        candidate_words = random.sample(self.WORDS, length * 2)
        word_pairs = []
        for i in range(0, length * 2, 2):
            word_pairs.append(f"{candidate_words[i]}: {candidate_words[i+1]}")

        return self.trim_context(", ".join(word_pairs), length)

    def generate_gibberish_words(self, length):
        words = []
        for _ in range(length):
            word_length = random.randint(2, 9)
            word = "".join(random.choices(string.ascii_lowercase, k=word_length))
            words.append(word)

        return self.trim_context(", ".join(words), length)


if __name__ == "__main__":
    generator = ContextGenerator()
    data = generator.generate_context("gibberish")
    print(data)
