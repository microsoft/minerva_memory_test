import random

from task.base_task import Task
from .context_utils import ContextGenerator


class CompareTwoLists(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "compare_two_lists"
        self.task_category = "spot_the_differences"

        self.num_samples = 10

        self.task_instruction = "There are two lists of words in the context. The first list contains the original words. The second list is similar to the first but has some words replaced with different ones. Your task is to identify the words in the {chosen_list} list that are different from those in the {other_list} list. Provide the different words as your answer."

        self.variables = {
            "context_length": [2000],
            "n_difference": [1, 5, 10, 20],
            "chosen_list": ["first", "second"],
        }

        self.metrics = ["exact_match", "rouge"]

    def format_prompt(self, context, chosen_list):
        if chosen_list == "first":
            other_list = "second"
        else:
            other_list = "first"

        task_instruction = self.task_instruction.format(
            chosen_list=chosen_list, other_list=other_list
        )
        return (
            "Context:\n"
            + context
            + "\n\nInstruction:\n"
            + task_instruction
            + "\n\nAnswer:"
        )

    def replace_words(self, context, n_difference):
        words = context.split(", ")
        indices = random.sample(range(len(words)), n_difference)
        original_words = [words[i] for i in indices]
        replacing_words = random.sample(self.WORDS, n_difference)
        for i in range(n_difference):
            words[indices[i]] = replacing_words[i]

        new_context = ", ".join(words)

        updated_context = "List 1: " + context + "\nList 2: " + new_context

        return updated_context, original_words, replacing_words

    def compile_test_entry(
        self,
        context,
        different_words_first,
        different_words_second,
        chosen_list,
        n_difference,
        length,
    ):

        entry_id = self.create_entry_id()
        prompt = self.format_prompt(context, chosen_list)
        if chosen_list == "first":
            different_words = different_words_first
        else:
            different_words = different_words_second
        reference = ", ".join(different_words)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "level": self.task_category,
            "task": self.task_name,
            "context_length": 2 * length,
            "n_difference": n_difference,
            "chosen_list": chosen_list,
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
                for n_difference in self.variables["n_difference"]:
                    updated_context, different_words_first, different_words_second = (
                        self.replace_words(context, n_difference)
                    )
                    for chosen_list in self.variables["chosen_list"]:
                        entry = self.compile_test_entry(
                            updated_context,
                            different_words_first,
                            different_words_second,
                            chosen_list,
                            n_difference,
                            length,
                        )
                        self.task_data.append(entry)

        return self.task_data


class IdentifyOddGroup(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "identify_the_odd_group"
        self.task_category = "spot_the_differences"

        self.num_samples = 5

        self.task_instruction = 'Given the lists of words in the context, identify the list that is different from the others. Provide the list number as your answer. For example, if the Nth list is different, provide "List N" as your answer.'

        self.variables = {
            "context_length": [4000],
            "n_words": [25, 50, 75, 100],
            "p_anomaly": [0, 0.25, 0.5],
        }

        self.metrics = ["exact_match"]

    def create_context_data(self, n_words, context_length):
        selected_words = random.sample(self.WORDS, n_words)
        list_token_length = ContextGenerator.get_context_length(
            "List 1: " + ", ".join(selected_words) + "\n"
        )

        n_list = context_length // list_token_length

        context = []
        for _ in range(n_list):
            permutation = random.sample(selected_words, n_words)
            context.append(", ".join(permutation))

        return context

    def format_prompt(self, context):
        return (
            "Context:\n"
            + context
            + "\n\nInstruction:\n"
            + self.task_instruction
            + "\n\nAnswer:"
        )

    def sample_anomaly(self, context, n_words, p_anomaly):
        n_list = len(context)
        if p_anomaly == 0:
            n_anomaly = 1
        else:
            n_anomaly = int(n_words * p_anomaly)

        anomaly_list_index = random.choice(range(n_list))
        corrupted_list = context[anomaly_list_index].split(", ")
        corrupted_indices = random.sample(range(n_words), n_anomaly)
        for i in corrupted_indices:
            corrupted_list[i] = random.choice(self.WORDS)

        context[anomaly_list_index] = ", ".join(corrupted_list)

        context = [f"List {i+1}: {context[i]}" for i in range(n_list)]
        context = "\n".join(context)

        return context, anomaly_list_index

    def compile_test_entry(self, context, n_words, context_length, p_anomaly):
        entry_id = self.create_entry_id()
        context, anomaly_list_index = self.sample_anomaly(context, n_words, p_anomaly)
        prompt = self.format_prompt(context)
        reference = "List " + str(anomaly_list_index + 1)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": context_length,
            "n_words": n_words,
            "p_anomaly": p_anomaly,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            for _ in range(self.num_samples):
                for n_words in self.variables["n_words"]:
                    context = self.create_context_data(n_words, length)
                    for p_anomaly in self.variables["p_anomaly"]:
                        entry = self.compile_test_entry(
                            context, n_words, length, p_anomaly
                        )
                        self.task_data.append(entry)

        return self.task_data


class PatchDifference(Task):
    def __init__(self):
        super().__init__()
        self.task_name = "patch_the_difference"
        self.task_category = "spot_the_differences"

        self.num_samples = 5

        self.task_instruction = "Given the sequence of words that follows a specific pattern in the context, predict the {nth} word that appears after the final word in the given sequence.\n\nAnswer: The {nth} word that appears after the final word in the given sequence is"

        self.variables = {
            "context_length": [4000],
            "pattern_length": [2, 15, 30],
            "start": [0, 0.5, 1],
            "nth": [1, 3, 6],
        }

        self.metrics = ["exact_match"]

    def create_context_data(self, context_length, pattern_length):
        selected_words = random.sample(self.WORDS, pattern_length)
        pattern = ", ".join(selected_words) + ", "

        pattern_token_length = ContextGenerator.get_context_length(pattern)
        context = pattern * (context_length // pattern_token_length)

        context = context.rstrip(", ")

        return context, selected_words

    def format_prompt(self, context, nth):
        if nth == 1:
            nth = "next"
        elif nth == 2:
            nth = "second"
        elif nth == 3:
            nth = "third"
        else:
            nth = f"{nth}th"

        instruction = self.task_instruction.format(nth=nth)
        return "Context:\n" + context + "\n\nInstruction:\n" + instruction

    def compile_test_entry(
        self, context, pattern_words, start, nth, context_length, pattern_length
    ):
        if start == 0:
            additional_words = [pattern_words[0]]
            start_index = 0
        elif start == 1:
            additional_words = []
            start_index = len(pattern_words) - 1
        else:
            additional_words = pattern_words[: int(start * len(pattern_words))]
            start_index = len(additional_words) - 1

        if additional_words:
            context = context + ", " + ", ".join(additional_words)
        prompt = self.format_prompt(context, nth)

        nth_index = (start_index + nth) % len(pattern_words)
        reference = pattern_words[nth_index]

        entry_id = self.create_entry_id()

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": context_length,
            "pattern_length": pattern_length,
            "start": start,
            "nth": nth,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            for _ in range(self.num_samples):
                for pattern_length in self.variables["pattern_length"]:
                    context, pattern_words = self.create_context_data(
                        length, pattern_length
                    )
                    for start in self.variables["start"]:
                        if pattern_length < 3 and start == 0.5:
                            continue
                        for nth in self.variables["nth"]:
                            entry = self.compile_test_entry(
                                context,
                                pattern_words,
                                start,
                                nth,
                                length,
                                pattern_length,
                            )
                            self.task_data.append(entry)

        return self.task_data


