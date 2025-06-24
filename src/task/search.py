import random

from task.base_task import Task


class StringSearchWord(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "string_search_word"
        self.task_category = "search"
        self.num_samples = 5

        self.context_type = "unique_words"
        self.task_instruction = "Given the context, determine if the word \"{query_word}\" is present in the context. Answer with 'yes' or 'no'."

        self.variables = {
            "context_length": [4000],
            "context_depth": [0, 0.25, 0.5, 0.75, 1],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, context, query_word):
        instruction = self.task_instruction.format(query_word=query_word)
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def sample_query_word(self, context, depth, label):
        context_words = context.split(", ")
        if label == "no":
            words = random.sample(self.WORDS, 100)
            for i in range(100):
                word = words[i]
                if word not in context_words:
                    return word
            return "nft"

        if depth == 1.0:
            return context_words[-1]
        else:
            context_len = len(context_words)
            return context_words[int(context_len * depth)]

    def compile_test_entry(self, context, length, depth, label):
        entry_id = self.create_entry_id()
        query_word = self.sample_query_word(context, depth, label)
        prompt = self.format_prompt(context, query_word)
        reference = label

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "context_depth": depth,
        }

        return entry

    def compile_task_data(self, context_type="unique_words"):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type=context_type, length=length, num_samples=self.num_samples
            )
            for context in context_data:
                for depth in self.variables["context_depth"]:
                    for label in ["yes", "no"]:
                        entry = self.compile_test_entry(context, length, depth, label)
                        self.task_data.append(entry)

        return self.task_data


class StringSearchGibberish(StringSearchWord):
    def __init__(self):
        super().__init__()
        self.task_name = "string_search_gibberish"

    def compile_task_data(self, context_type="gibberish"):
        return super().compile_task_data(context_type)
    

class StringSearchSequence(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "string_search_sequence"
        self.task_category = "search"
        self.num_samples = 10

        self.context_type = "unique_words"
        self.task_instruction = "Given the list of words in the context, determine if the sequence \"{query_sequence}\" appears in the context. Answer with 'yes' or 'no'."

        self.variables = {
            "context_length": [4000],
            "sequence_length": [8, 16, 32, 64],
            "n_corrupt": [1],
        }

        self.metrics = ["exact_match"]

    def sample_query_sequence(self, context, sequence_length):
        context_words = context.split(", ")

        subsequence_start = random.randint(0, len(context_words) - sequence_length)
        subsequence = context_words[
            subsequence_start : subsequence_start + sequence_length
        ]

        return subsequence

    def corrupt_sequence(self, subsequence, n_corrupt):
        corrupted_indices = random.sample(range(len(subsequence)), n_corrupt)
        for i in corrupted_indices:
            subsequence[i] = random.choice(self.WORDS)

        return subsequence

    def format_prompt(self, context, query_sequence):
        instruction = self.task_instruction.format(query_sequence=query_sequence)
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def compile_test_entry(
        self, context, context_length, subsequence, n_corrupt, label
    ):
        entry_id = self.create_entry_id()
        if label == "no":
            query_sequence = self.corrupt_sequence(subsequence, n_corrupt)
        else:
            query_sequence = subsequence
        query_sequence = ", ".join(query_sequence)
        prompt = self.format_prompt(context, query_sequence)
        reference = label

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": context_length,
            "sequence_length": len(subsequence),
            "n_corrupt": n_corrupt,
        }

        return entry

    def compile_task_data(self):
        for context_length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="unique_words",
                length=context_length,
                num_samples=self.num_samples,
            )
            for context in context_data:
                for sequence_length in self.variables["sequence_length"]:
                    for n_corrupt in self.variables["n_corrupt"]:
                        subsequence = self.sample_query_sequence(
                            context, sequence_length
                        )
                        for label in ["yes", "no"]:
                            entry = self.compile_test_entry(
                                context,
                                context_length,
                                subsequence,
                                n_corrupt,
                                label,
                            )
                            self.task_data.append(entry)

        return self.task_data


class KeyValueSearch(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "key_value_search"
        self.task_category = "search"
        self.context_type = "word_pairs"
        self.num_samples = 10

        self.task_instruction = 'Given a list of word pairs formatted as "word_1: word_2" in the context, return the second word associated with the provided first word. For the first word "{query_item}", the corresponding second word is:'

        self.variables = {
            "context_length": [4000],
            "context_depth": [0, 0.25, 0.5, 0.75, 1],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, context, query_item):
        instruction = self.task_instruction.format(query_item=query_item)
        return "Context:\n" + context + "\n\nInstruction:\n" + instruction

    def get_query_item(self, context, depth):
        word_pairs = context.split(", ")
        length = len(word_pairs)

        if depth == 1.0:
            index = -1
        else:
            index = int(length * depth)

        selected_pair = word_pairs[index]
        query_item = selected_pair.split(":")[0].strip()
        reference = selected_pair.split(":")[1].strip()

        return query_item, reference

    def compile_test_entry(self, context, length, depth):
        entry_id = self.create_entry_id()

        query_item, reference = self.get_query_item(context, depth)
        prompt = self.format_prompt(context, query_item)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "context_depth": depth,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="word_pairs",
                length=length,
                num_samples=self.num_samples,
            )
            for context in context_data:
                for depth in self.variables["context_depth"]:
                    entry = self.compile_test_entry(context, length, depth)
                    self.task_data.append(entry)

        return self.task_data


class BatchKeyValueSearch(KeyValueSearch):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "batch_key_value_search"
        self.task_category = "search"
        self.num_samples = 5

        self.task_instruction = 'Given a list of word pairs formatted as "word_1: word_2" in the context, return the second words associated with the provided first words. For the first words "{query_item}", the corresponding second words are:'

        self.variables = {
            "context_length": [4000],
            "n_words": [4, 8, 16, 32],
        }

        self.metrics = ["exact_match", "rouge"]

    def format_prompt(self, context, query_item):
        instruction = self.task_instruction.format(query_item=query_item)
        return "Context:\n" + context + "\n\nInstruction:\n" + instruction

    def get_query_item(self, context, n_words):
        word_pairs = context.split(", ")
        selected_indices = [
            int(len(word_pairs) / (n_words - 1) * i) for i in range(n_words)
        ]
        selected_indices[-1] = len(word_pairs) - 1

        selected_pairs = [word_pairs[index] for index in selected_indices]
        query_item = ", ".join([pair.split(":")[0].strip() for pair in selected_pairs])
        reference = ", ".join([pair.split(":")[1].strip() for pair in selected_pairs])

        return query_item, reference

    def compile_test_entry(self, context, length, n_words):
        entry_id = self.create_entry_id()
        query_items, reference = self.get_query_item(context, n_words)
        prompt = self.format_prompt(context, query_items)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "n_words": n_words,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="word_pairs",
                length=length,
                num_samples=self.num_samples,
            )
            for context in context_data:
                for n_words in self.variables["n_words"]:
                    entry = self.compile_test_entry(context, length, n_words)
                    self.task_data.append(entry)

        return self.task_data

