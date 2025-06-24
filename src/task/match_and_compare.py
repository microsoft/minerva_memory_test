import random

from .context_utils import ContextGenerator
from task.base_task import Task


class ComparePositions(Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_name = "compare_positions"
        self.task_category = "match_and_compare"

        self.num_samples = 3

        self.task_instruction = 'Given the list of words in the context, determine the relative positions of two words. Does the word "{word_1}" come before the word "{word_2}" in the list? Answer "yes" or "no".'

        self.variables = {
            "context_length": [4000],
            "context_depth_1": [0, 0.25, 0.5, 0.75, 1],
            "context_depth_2": [0, 0.25, 0.5, 0.75, 1],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, context, word_1, word_2):
        instruction = self.task_instruction.format(word_1=word_1, word_2=word_2)
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def sample_words(self, context, depth_1, depth_2):
        context_data = context.split(", ")
        length = len(context_data)
        if depth_1 == 1.0:
            index_1 = length - 1
        else:
            index_1 = int(length * depth_1)

        if depth_2 == 1.0:
            index_2 = length - 1
        else:
            index_2 = int(length * depth_2)

        if depth_1 == depth_2:
            if index_2 == length - 1:
                index_1 = length - 2
            else:
                index_2 = index_1 + 1

        word_1 = context_data[index_1]
        word_2 = context_data[index_2]

        return word_1, word_2

    def get_reference(self, depth_1, depth_2):
        if depth_1 <= depth_2:
            return "yes"
        else:
            return "no"

    def compile_test_entry(self, context, length, depth_1, depth_2):
        entry_id = self.create_entry_id()
        word_1, word_2 = self.sample_words(context, depth_1, depth_2)
        reference = self.get_reference(depth_1, depth_2)

        prompt = self.format_prompt(context, word_1, word_2)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "context_depth_1": depth_1,
            "context_depth_2": depth_2,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="unique_words", length=length, num_samples=self.num_samples
            )
            for context in context_data:
                for depth_1 in self.variables["context_depth_1"]:
                    for depth_2 in self.variables["context_depth_2"]:
                        entry = self.compile_test_entry(
                            context, length, depth_1, depth_2
                        )
                        self.task_data.append(entry)

        return self.task_data


class FindDuplicates(Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_name = "find_duplicates"
        self.task_category = "match_and_compare"
        self.num_samples = 5

        self.task_instruction = "A word is repeated multiple times in the context. Your task is to identify the word that is repeated.\n\nThe repeated word is:"

        self.variables = {
            "context_length": [4000],
            "repetition_count": [2, 4, 8, 16, 32],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, context_str):
        return "Context:\n" + context_str + "\n\nInstruction:\n" + self.task_instruction

    def create_repeated_context(self, context, repetition_count):
        new_context = context.split(", ")
        length = len(new_context)
        index = random.choice(range(length))
        repeated_word = new_context[index]

        if index == 0:
            index_range = range(1, length)
        elif index == length - 1:
            index_range = range(length - 1)
        else:
            index_range = list(range(index)) + list(range(index + 1, length))
        indices = random.sample(index_range, repetition_count - 1)
        for i in indices:
            new_context[i] = repeated_word
        return ", ".join(new_context), repeated_word

    def compile_test_entry(self, context, length, repetition_count):
        entry_id = self.create_entry_id()
        repeated_context, reference = self.create_repeated_context(
            context, repetition_count
        )
        prompt = self.format_prompt(repeated_context)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "repetition_count": repetition_count,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="unique_words",
                length=length,
                num_samples=self.num_samples,
            )
            for context in context_data:
                for repetition_count in self.variables["repetition_count"]:
                    entry = self.compile_test_entry(context, length, repetition_count)
                    self.task_data.append(entry)

        return self.task_data


class Count(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "count"
        self.task_category = "match_and_compare"
        self.num_samples = 5

        self.task_instruction = 'Count the number of times the word "{repeated_word}" appears in the context.\n\nAnswer: The word "{repeated_word}" appears'

        self.variables = {
            "context_length": [4000],
            "repetition_count": [2, 4, 8, 16, 32],
        }

        self.metrics = ["exact_match", "count_accuracy"]

    def format_prompt(self, context_str, repeated_word):
        instruction = self.task_instruction.format(repeated_word=repeated_word)
        return "Context:\n" + context_str + "\n\nInstruction:\n" + instruction

    def create_repeated_context(self, context, repetition_count):
        new_context = context.split(", ")
        length = len(new_context)
        index = random.choice(range(length))
        repeated_word = new_context[index]

        if index == 0:
            index_range = range(1, length)
        elif index == length - 1:
            index_range = range(length - 1)
        else:
            index_range = list(range(index)) + list(range(index + 1, length))
        indices = random.sample(index_range, repetition_count - 1)
        for i in indices:
            new_context[i] = repeated_word
        return ", ".join(new_context), repeated_word

    def compile_test_entry(self, context, length, repetition_count):
        entry_id = self.create_entry_id()
        repeated_context, repeated_word = self.create_repeated_context(
            context, repetition_count
        )
        prompt = self.format_prompt(repeated_context, repeated_word)
        reference = str(repetition_count)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "repetition_count": repetition_count,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="unique_words",
                length=length,
                num_samples=self.num_samples,
            )
            for context in context_data:
                for repetition_count in self.variables["repetition_count"]:
                    entry = self.compile_test_entry(context, length, repetition_count)
                    self.task_data.append(entry)

        return self.task_data


class CheckAssociation(Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_name = "check_association"
        self.task_category = "match_and_compare"
        self.num_samples = 5

        self.task_instruction = 'Given the context with words and their assigned attributes in the format of "word: ATT_N", determine if the word "{query_word}" has the same attribute as the word "{reference_word}"? Answer "yes" or "no".'
        self.variables = {
            "context_length": [4000],
            "n_attribute": [2, 4, 8, 16, 32],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, context, query_word, reference_word):
        instruction = self.task_instruction.format(
            query_word=query_word, reference_word=reference_word
        )
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def create_context_data(self, n_attribute, length=4096):
        words = random.sample(self.WORDS, length)
        attribute_words = ["ATT_" + str(i) for i in range(1, n_attribute + 1)]
        context = []
        for word in words:
            attribute = random.choice(attribute_words[:n_attribute])
            context.append(f"{word}: {attribute}")

        context = ", ".join(context)
        context = ContextGenerator.trim_context(context, length)

        return context

    def sample_query_words(self, context, label):
        context_data = context.split(", ")
        attribute_dict = {}
        for item in context_data:
            word, attribute = item.split(": ")
            attribute_dict[attribute] = attribute_dict.get(attribute, []) + [word]

        if label == "yes":
            attribute = random.choice(list(attribute_dict.keys()))
            query_word, reference_word = random.sample(attribute_dict[attribute], 2)

        else:
            attribute_1, attribute_2 = random.sample(list(attribute_dict.keys()), 2)
            query_word = random.choice(attribute_dict[attribute_1])
            reference_word = random.choice(attribute_dict[attribute_2])

        return query_word, reference_word

    def compile_test_entry(self, context, length, n_attribute, label):
        entry_id = self.create_entry_id()
        query_word, reference_word = self.sample_query_words(context, label)
        prompt = self.format_prompt(context, query_word, reference_word)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": label,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "n_attribute": n_attribute,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            for n_attribute in self.variables["n_attribute"]:
                for _ in range(self.num_samples):
                    context = self.create_context_data(
                        n_attribute=n_attribute, length=length
                    )
                    for label in ["yes", "no"]:
                        entry = self.compile_test_entry(
                            context, length, n_attribute, label
                        )
                        self.task_data.append(entry)

        return self.task_data
