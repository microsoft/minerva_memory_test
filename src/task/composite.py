import random

from task.base_task import Task


class ProcessingDataBlocks(Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_name = "processing_data_blocks"
        self.task_category = "composite"
        self.num_samples = 10

        self.task_instruction = 'The context consists of a series of alternating roles, each associated with a list of words. Your task is to identify and recall all the words from the role labeled "{query_role}" that appear after the word "{query_word}" in the sequence. Please write your answer after the text "Answer:". For example, "Answer: word1, word2, word3".'
        self.variables = {
            "context_length": [4000],
            "n_roles": [2, 4, 8, 16, 32],
            "n_turns": [10],
        }

        self.metrics = ["exact_match", "rouge"]

    def format_prompt(self, roles, query_role, query_word):
        n_turns = len(roles["Role 1"])
        context = [[] for _ in range(n_turns)]
        for role_name, role_words in roles.items():
            for i in range(n_turns):
                context[i].append(f"{role_name}: {', '.join(role_words[i])}")

        context = "\n".join(["\n".join(segment) for segment in context])

        instruction = self.task_instruction.format(
            query_role=query_role, query_word=query_word
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

    def sample_query_role(self, formatted_roles, n_roles, n_turns):
        query_role = random.randint(0, n_roles - 1)
        query_role = f"Role {query_role + 1}"
        query_turn = random.randint(0, n_turns - 1)
        query_word_index = random.randint(
            0, len(formatted_roles[query_role][query_turn]) - 1
        )
        query_word = formatted_roles[query_role][query_turn][query_word_index]

        reference_words = formatted_roles[query_role][query_turn][
            query_word_index + 1 :
        ]
        for i in range(query_turn + 1, n_turns):
            reference_words += formatted_roles[query_role][i]
        reference = ", ".join(reference_words)

        return query_role, query_word, reference

    def compile_test_entry(self, roles, length, n_roles, n_turns):
        entry_id = self.create_entry_id()
        query_role, query_word, reference = self.sample_query_role(
            roles, n_roles, n_turns
        )
        prompt = self.format_prompt(roles, query_role, query_word)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": reference,
            "category": self.task_category,
            "task": self.task_name,
            "context_length": length,
            "n_roles": n_roles,
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
                        entry = self.compile_test_entry(roles, length, n_roles, n_turns)
                        self.task_data.append(entry)

        return self.task_data


class TheoryOfMind(Task):
    def __init__(self) -> None:
        super().__init__()
        self.task_name = "theory_of_mind"
        self.task_category = "composite"
        self.num_samples = 10

        self.task_instruction = 'Given the actions of the agents, your task is to determine the final list of words each agent ends up with after a series of actions. Write your final answer after the text "FINAL ANSWER:". For example, "FINAL ANSWER: Agent A: word1, word2, word3\nAgent B: word4, word5".'

        self.variables = {
            "num_agents": [2, 3, 4],
            "action_step": [100, 200],
            "state_size": [10],
        }

        self.metrics = ["exact_match", "rouge", "theory_of_mind"]

    def format_prompt(self, agent_actions):
        prompt = (
            "Agents actions:\n\n"
            + agent_actions
            + "\n\nInstruction:\n"
            + self.task_instruction
            + "\n\nFINAL ANSWER:"
        )

        return prompt

    def create_context_data(self, num_agents, state_size, step):
        unique_words = random.sample(self.WORDS, state_size * 200)

        agent_names = [f"Agent {chr(65 + i)}" for i in range(num_agents)]
        agent_states = [[] for _ in range(num_agents)]
        all_actions = ""

        actions = ["draw", "discard", "swap"]

        # Initialize the agents' states
        for i in range(num_agents):
            n_words = random.randint(1, state_size - 1)
            initial_state = random.sample(unique_words, n_words)
            agent_states[i] = initial_state
            all_actions += f"{agent_names[i]} starts with the following words: {', '.join(initial_state)}.\n"

        i = 0
        while i < step:
            sampled_action = random.choice(actions)
            if sampled_action == "swap":
                sampled_agents = random.sample(list(range(num_agents)), 2)
            else:
                sampled_agents = random.sample(list(range(num_agents)), 1)

            if sampled_action == "draw":
                max_words = state_size - len(agent_states[sampled_agents[0]])
                if max_words == 0:
                    continue
                n_words = random.randint(1, max_words)
                available_words = [
                    word
                    for word in unique_words
                    if word not in agent_states[sampled_agents[0]]
                ]
                words = random.sample(available_words, n_words)
                agent_states[sampled_agents[0]] += words
                all_actions += f"{agent_names[sampled_agents[0]]} draws the following words: {', '.join(words)}.\n"

            elif sampled_action == "discard":
                max_words = int(len(agent_states[sampled_agents[0]]) / 2)
                if max_words == 0:
                    continue
                n_words = random.randint(1, max_words)
                words_to_discard = random.sample(
                    agent_states[sampled_agents[0]], n_words
                )
                agent_states[sampled_agents[0]] = [
                    word
                    for word in agent_states[sampled_agents[0]]
                    if word not in words_to_discard
                ]

                all_actions += f"{agent_names[sampled_agents[0]]} discards the following words: {', '.join(words_to_discard)}.\n"

            elif sampled_action == "swap":
                max_words = min(
                    len(agent_states[sampled_agents[0]]),
                    len(agent_states[sampled_agents[1]]),
                )
                max_words = max_words // 2
                if max_words == 0:
                    continue
                n_words = random.randint(1, max_words)

                words_to_swap_agent_1 = random.sample(
                    agent_states[sampled_agents[0]], n_words
                )
                words_to_swap_agent_2 = random.sample(
                    agent_states[sampled_agents[1]], n_words
                )

                agent_states[sampled_agents[0]] = [
                    word
                    for word in agent_states[sampled_agents[0]]
                    if word not in words_to_swap_agent_1
                ]
                agent_states[sampled_agents[0]] += words_to_swap_agent_2

                agent_states[sampled_agents[1]] = [
                    word
                    for word in agent_states[sampled_agents[1]]
                    if word not in words_to_swap_agent_2
                ]
                agent_states[sampled_agents[1]] += words_to_swap_agent_1

                all_actions += f"{agent_names[sampled_agents[0]]} swaps the following words \"{', '.join(words_to_swap_agent_1)}\" with {agent_names[sampled_agents[1]]} for the following words \"{', '.join(words_to_swap_agent_2)}\".\n"

            i += 1

        agent_final_states = {
            f"Agent {chr(65 + i)}": agent_states[i] for i in range(num_agents)
        }

        return all_actions, agent_final_states

    def compile_test_entry(
        self, agent_actions, agent_final_states, action_step, state_size
    ):
        entry_id = self.create_entry_id()
        prompt = self.format_prompt(agent_actions)

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": agent_final_states,
            "category": self.task_category,
            "task": self.task_name,
            "num_agents": len(agent_final_states),
            "step": action_step,
            "state_size": state_size,
        }

        return entry

    def compile_task_data(self):
        for num_agents in self.variables["num_agents"]:
            for step in self.variables["action_step"]:
                for state_size in self.variables["state_size"]:
                    for _ in range(self.num_samples):
                        agent_actions, agent_final_states = self.create_context_data(
                            num_agents, state_size, step
                        )
                        entry = self.compile_test_entry(
                            agent_actions, agent_final_states, step, state_size
                        )
                        self.task_data.append(entry)

        return self.task_data
