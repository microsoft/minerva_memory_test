import random

from task.base_task import Task


class Snapshot(Task):
    def __init__(self, context_type="unique_words") -> None:
        super().__init__()
        self.task_category = "recall_and_edit"

        self.context_type = context_type
        self.task_name = "snapshot" + "_" + context_type

        self.num_samples = 10

        self.task_instruction = "Repeat the previous context exactly as it is, without making any additions or deletions."

        # variables for the task
        self.variables = {"context_length": [4000]}

        self.metrics = ["exact_match", "rouge"]

    def format_prompt(self, context):
        return (
            "Context:\n"
            + context
            + "\n\nInstruction:\n"
            + self.task_instruction
            + "\n\nAnswer:"
        )

    def compile_test_entry(self, context, length):
        entry_id = self.create_entry_id()
        prompt = self.format_prompt(context)
        reference = self.get_reference(context)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type=self.context_type,
                length=length,
                num_samples=self.num_samples,
            )
            for context in context_data:
                entry = self.compile_test_entry(context, length)
                self.task_data.append(entry)

        return self.task_data

    
    def get_reference(self, context):
        return context


class ReplaceAll(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "replace_all"
        self.task_category = "recall_and_edit"
        self.num_samples = 5

        self.task_instruction = 'Repeat the previous context and replace the word "{query_item}" with "{substitute}" each time it appears.'

        self.variables = {
            "context_length": [4000],
            "density": [0.2, 0.4, 0.6, 0.8],
        }

        self.metrics = ["exact_match", "rouge"]

    def format_prompt(self, context_str, query_item, substitute):
        instruction = self.task_instruction.format(
            query_item=query_item, substitute=substitute
        )
        return (
            "Context:\n"
            + context_str
            + "\n\nInstruction:\n"
            + instruction
            + "\n\nAnswer:"
        )

    def create_context_with_repeated_item(self, context, density):
        query_item, substitute = random.sample(self.WORDS, 2)
        new_context = context.split(", ")
        num_repetition = int(len(new_context) * density)
        indices = random.sample(range(len(new_context)), num_repetition)
        for i in indices:
            new_context[i] = query_item
        context_str = ", ".join(new_context)

        for i in indices:
            new_context[i] = substitute
        reference = ", ".join(new_context)

        return context_str, reference, query_item, substitute

    def compile_test_entry(self, context, length, density):
        entry_id = self.create_entry_id()

        context_str, reference, query_item, substitute = (
            self.create_context_with_repeated_item(context, density)
        )
        prompt = self.format_prompt(context_str, query_item, substitute)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "density": density,
        }
        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="unique_words", length=length, num_samples=self.num_samples
            )
            for context in context_data:
                for density in self.variables["density"]:
                    entry = self.compile_test_entry(context, length, density)
                    self.task_data.append(entry)

        return self.task_data


class ReplaceAllXToNull(ReplaceAll):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "replace_all_x_to_null"
        self.task_category = "recall_and_edit"
        self.num_samples = 5

        self.task_instruction = 'Repeat the previous context but skip the word "{query_item}" each time it appears.'

    def format_prompt(self, context_str, query_item):
        instruction = self.task_instruction.format(query_item=query_item)
        return (
            "Context:\n"
            + context_str
            + "\n\nInstruction:\n"
            + instruction
            + "\n\nAnswer:"
        )

    def create_context_with_repeated_item(self, context, density):
        query_item = random.choice(self.WORDS)
        new_context = context.split(", ")
        num_repetition = int(len(new_context) * density)
        indices = random.sample(range(len(new_context)), num_repetition)
        for i in indices:
            new_context[i] = query_item
        context_str = ", ".join(new_context)

        for i in indices:
            new_context[i] = ""
        reference = [item for item in new_context if item != ""]
        reference = ", ".join(reference)

        return context_str, reference, query_item

    def compile_test_entry(self, context, length, density):
        entry_id = self.create_entry_id()

        context_str, reference, query_item = self.create_context_with_repeated_item(
            context, density
        )
        prompt = self.format_prompt(context_str, query_item)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "density": density,
        }
        return entry


class OverwritePositions(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "overwrite_positions"
        self.task_category = "recall_and_edit"
        self.num_samples = 5

        self.task_instruction = 'Repeat the previous context and replace every {nth} word with "{substitute}".'

        self.variables = {
            "context_length": [4000],
            "nth": [2, 3, 4],
        }

        self.metrics = ["exact_match", "rouge"]


    def format_prompt(self, context, nth, substitute):
        if nth == 2:
            nth = "other"
        elif nth == 3:
            nth = "third"
        elif nth == 4:
            nth = "fourth"
        instruction = self.task_instruction.format(nth=nth, substitute=substitute)
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def get_reference(self, context, nth, substitute):
        reference = context.split(", ")
        for i in range(nth - 1, len(reference), nth):
            reference[i] = substitute
        return ", ".join(reference)

    def compile_test_entry(self, context, length, nth):
        entry_id = self.create_entry_id()
        substitute = random.choice(self.WORDS)
        reference = self.get_reference(context, nth, substitute)

        prompt = self.format_prompt(context, nth, substitute)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "nth": nth,
        }
        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="unique_words", length=length, num_samples=self.num_samples
            )
            for context in context_data:
                for nth in self.variables["nth"]:
                    entry = self.compile_test_entry(context, length, nth)
                    self.task_data.append(entry)

        return self.task_data


class OverwritePositionsNthToNull(OverwritePositions):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "overwrite_positions_nth_to_null"
        self.task_category = "recall_and_edit"
        self.num_samples = 5

        self.task_instruction = "Repeat the previous context but skip every {nth} word."

    def format_prompt(self, context, nth):
        if nth == 2:
            nth = "other"
        elif nth == 3:
            nth = "third"
        elif nth == 4:
            nth = "fourth"
        instruction = self.task_instruction.format(nth=nth)
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def get_reference(self, context, nth):
        reference = context.split(", ")
        for i in range(nth - 1, len(reference), nth):
            reference[i] = ""
        reference = [item for item in reference if item != ""]
        return ", ".join(reference)

    def compile_test_entry(self, context, length, nth):
        entry_id = self.create_entry_id()
        reference = self.get_reference(context, nth)

        prompt = self.format_prompt(context, nth)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "nth": nth,
        }
        return entry


class FunctionalUpdates(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "functional_updates"
        self.task_category = "recall_and_edit"
        self.num_samples = 5

        self.task_instruction = ""

        self.metrics = ["exact_match", "rouge"]

        self.variables = {
            "context_length": [4000],
            "operation": ["add", "subtract", "multiply"],
        }

    def format_prompt(self, context, operation):
        if operation == "add":
            # hardcoded value for simplicity, can be parameterized if needed
            self.task_instruction = (
                "Add 3 to every number in the previous context."
            )
        elif operation == "subtract":
            self.task_instruction = (
                "Subtract 1 from every number in the previous context."
            )
        elif operation == "multiply":
            self.task_instruction = (
                "Multiply every number in the previous context by 2."
            )

        return (
            "Context:\n"
            + context
            + "\n\nInstruction:\n"
            + self.task_instruction
            + "\n\nAnswer:"
        )

    def get_reference(self, context, operation):
        context_data = context.split(", ")
        if operation == "add":
            new_context = [int(x) + 3 for x in context_data]
        elif operation == "subtract":
            new_context = [int(x) - 1 for x in context_data]
        elif operation == "multiply":
            new_context = [int(x) * 2 for x in context_data]
        return ", ".join([str(x) for x in new_context])

    def compile_test_entry(self, context, length, operation):
        entry_id = self.create_entry_id()
        prompt = self.format_prompt(context, operation)
        reference = self.get_reference(context, operation)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "operation": operation,
        }

        return entry

    def compile_task_data(self):
        for length in self.variables["context_length"]:
            context_data = self.create_context_data(
                context_type="random_numbers",
                length=length,
                num_samples=self.num_samples,
            )
            for context in context_data:
                for operation in self.variables["operation"]:
                    entry = self.compile_test_entry(context, length, operation)
                    self.task_data.append(entry)

        return self.task_data