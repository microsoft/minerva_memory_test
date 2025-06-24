import random

from task.base_task import Task


class GroupMembership(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "group_membership"
        self.task_category = "compute_on_sets_and_lists"
        self.num_samples = 5

        self.task_instruction = 'Given the lists of words in the context, determine which list contains the word "{query_word}". If the word is not present in any list, answer "no".'

        # variables for the task
        self.variables = {
            "context_length": [4000],
            "n_list": [4, 8, 16, 32],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, lists, query_word):
        context = ""
        for list_name, list_words in lists:
            context += f"{list_name}: {', '.join(list_words)}\n"

        instruction = self.task_instruction.format(query_word=query_word)
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def format_context(self, context, n_list):
        context_words = context.split(", ")
        lists = []
        for i in range(n_list):
            list_name = f"List {i+1}"
            list_words = context_words[i::n_list]
            lists.append((list_name, list_words))

        return lists

    def sample_query_word(self, context, lists, list_index):
        if list_index == len(lists):
            # If list_index is equal to the number of lists, we need to sample a word not in any list
            # Convert context to a set for O(1) membership checks
            context_words = set(context)
        
            # Try a few random samples first (likely to succeed quickly)
            for _ in range(20):
                candidate = random.choice(self.WORDS)
                if candidate not in context_words:
                    return candidate, "no"
        
            # If random sampling didn't work, find the first non-matching word
            for word in self.WORDS:
                if word not in context_words:
                    return word, "no"
                
            # Fallback if somehow all words are in context
            return "out_of_vocabulary_word", "no"

        # If list_index is within the range of lists, sample a word from that list
        random_index = random.randint(0, len(lists[list_index][1]) - 1)
        query_word = lists[list_index][1][random_index]
        reference = lists[list_index][0]

        return query_word, reference

    def compile_test_entry(self, context, lists, list_index, length):
        entry_id = self.create_entry_id()
        query_word, reference = self.sample_query_word(context, lists, list_index)
        prompt = self.format_prompt(lists, query_word)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "n_list": len(lists)
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
                for n_list in self.variables["n_list"]:
                    lists = self.format_context(context, n_list)
                    k = 4
                    sampled_list_indices = [int(i / k * n_list) for i in range(k + 1)]
                    for list_index in sampled_list_indices:
                        entry = self.compile_test_entry(
                            context, lists, list_index, length
                        )
                        self.task_data.append(entry)

        return self.task_data


class GroupAssociation(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "group_association"
        self.task_category = "compute_on_sets_and_lists"
        self.num_samples = 5

        self.task_instruction = 'Given the lists of words in the context, determine if the word "{query_word}" and the word "{reference_word}" are in the same list. Answer with "yes" or "no".'

        self.variables = {
            "context_length": [4000],
            "n_list": [4, 8, 16, 32],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, lists, query_word, reference_word):
        context = ""
        for list_name, list_words in lists:
            context += f"{list_name}: {', '.join(list_words)}\n"

        instruction = self.task_instruction.format(
            query_word=query_word, reference_word=reference_word
        )
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def format_context(self, context, n_list):
        context_words = context.split(", ")
        lists = []
        for i in range(n_list):
            list_name = f"List {i+1}"
            list_words = context_words[i::n_list]
            lists.append((list_name, list_words))

        return lists

    def sample_query_word(self, lists, label):
        n_list = len(lists)
        if label == "no":
            list_indices = random.sample(range(n_list), 2)
            query_word = random.choice(lists[list_indices[0]][1])
            reference_word = random.choice(lists[list_indices[1]][1])
        elif label == "yes":
            list_index = random.randint(0, n_list - 1)
            query_word, reference_word = random.sample(lists[list_index][1], 2)

        return query_word, reference_word

    def compile_test_entry(self, lists, length, label):
        entry_id = self.create_entry_id()
        query_word, reference_word = self.sample_query_word(lists, label)
        prompt = self.format_prompt(lists, query_word, reference_word)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": label,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "n_list": len(lists),
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
                for n_list in self.variables["n_list"]:
                    lists = self.format_context(context, n_list)
                    for label in ["yes", "no"]:
                        entry = self.compile_test_entry(lists, length, label)
                        self.task_data.append(entry)

        return self.task_data


class AlternatingGroupAssociation(Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_name = "alternating_group_association"
        self.task_category = "compute_on_sets_and_lists"
        self.num_samples = 5

        self.task_instruction = 'Given the context with alternating roles and their respective context words, determine if the word "{query_word}" and the word "{reference_word}" are in the same role. Answer with "yes" or "no".'
        self.variables = {
            "context_length": [4000],
            "n_roles": [2, 4, 8, 16, 32],
            "n_turns": [10],
        }

        self.metrics = ["exact_match"]

    def format_prompt(self, roles, query_word, reference_word):
        n_turns = len(roles["Role 1"])
        context = [[] for _ in range(n_turns)]
        for role_name, role_words in roles.items():
            for i in range(n_turns):
                context[i].append(f"{role_name}: {', '.join(role_words[i])}")

        context = "\n".join(["\n".join(segment) for segment in context])

        instruction = self.task_instruction.format(
            query_word=query_word, reference_word=reference_word
        )
        return (
            "Context:\n" + context + "\n\nInstruction:\n" + instruction + "\n\nAnswer:"
        )

    def format_context(self, context, n_roles, n_turns):
        context_words = context.split(", ")
        formatted_roles = {}
        role_length = len(context_words) // n_roles
        for i in range(n_roles):
            role_name = f"Role {i + 1}"
            role_words = context_words[i * role_length : (i + 1) * role_length]
            formatted_roles[role_name] = []
            for j in range(n_turns):
                segment_length = role_length // n_turns
                role_segment = role_words[j * segment_length : (j + 1) * segment_length]
                formatted_roles[role_name].append(role_segment)

        return formatted_roles

    def sample_query_word(self, formatted_roles, label):
        if label == "yes":
            selected_role = random.choice(list(formatted_roles.keys()))
            selected_turns = random.sample(formatted_roles[selected_role], 2)
            query_word = random.choice(selected_turns[0])
            reference_word = random.choice(selected_turns[1])
        else:
            selected_roles = random.sample(list(formatted_roles.keys()), 2)

            query_word = random.choice(
                random.choice(formatted_roles[selected_roles[0]])
            )
            reference_word = random.choice(
                random.choice(formatted_roles[selected_roles[1]])
            )
        return query_word, reference_word

    def compile_test_entry(self, roles, length, label, n_turns):
        entry_id = self.create_entry_id()
        query_word, reference_word = self.sample_query_word(roles, label)
        prompt = self.format_prompt(roles, query_word, reference_word)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": label,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "n_roles": len(roles),
            "n_turns": n_turns,
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
                for n_roles in self.variables["n_roles"]:
                    for n_turns in self.variables["n_turns"]:
                        roles = self.format_context(context, n_roles, n_turns)
                        for label in ["yes", "no"]:
                            entry = self.compile_test_entry(roles, length, label, n_turns)
                            self.task_data.append(entry)

        return self.task_data


class Iterate(Task):
    def __init__(self, word_index="last") -> None:
        super().__init__()

        self.task_category = "compute_on_sets_and_lists"
        self.task_name = f"iterate_{word_index}"

        self.num_samples = 5


        self.word_index = word_index

        self.task_instruction = f"Given the lists of words in the context, identify and recall the {self.word_index} word from each list. Provide your answer as a list of these words separated by commas."

        # variables for the task
        self.variables = {
            "context_length": [4000],
            "n_list": [4, 8, 16, 32],
        }

        self.metrics = ["exact_match", "rouge"]

    def format_prompt(self, lists):
        context = ""
        for list_name, list_words in lists:
            context += f"{list_name}: {', '.join(list_words)}\n"

        return (
            "Context:\n"
            + context
            + "\n\nInstruction:\n"
            + self.task_instruction
            + "\n\nAnswer:"
        )

    def format_context(self, context, n_list):
        context_words = context.split(", ")
        lists = []
        for i in range(n_list):
            list_name = f"List {i+1}"
            list_words = context_words[i::n_list]
            lists.append((list_name, list_words))

        return lists

    def get_reference(self, lists):
        reference = []
        if self.word_index == "first":
            word_index = 0
        elif self.word_index == "last":
            word_index = -1
        else:
            word_index = int(self.word_index)

        for list_name, list_words in lists:
            reference.append(list_words[word_index])

        return ", ".join(reference)

    def compile_test_entry(self, lists, length):
        entry_id = self.create_entry_id()
        prompt = self.format_prompt(lists)
        reference = self.get_reference(lists)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "n_list": len(lists),
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
                for n_list in self.variables["n_list"]:
                    lists = self.format_context(context, n_list)
                    entry = self.compile_test_entry(lists, length)
                    self.task_data.append(entry)

        return self.task_data

