import random

from task.base_task import Task


class QuantityState(Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_name = "quantity_state"
        self.task_category = "stateful_processing"
        self.num_samples = 10

        self.task_instruction = 'In the context, you are given an initial number and a series of operations to perform on that number. Your task is to determine the final result of the operations. Write your final answer after the text "FINAL ANSWER:". For example, "FINAL ANSWER: 42".'
        self.variables = {
            "operation_step": [200],
        }

        self.metrics = ["final_answer_exact_match"]

    def format_prompt(self, operations):
        prompt = (
            "Context:\n\n"
            + operations
            + "\n\nInstruction:\n"
            + self.task_instruction
            + "\n\nFINAL ANSWER:"
        )

        return prompt

    def create_context_data(self, step):
        initial_number = random.randint(1, 100)
        final_number = initial_number
        operations = (
            "Begin with the number "
            + str(initial_number)
            + ". Perform the following operations:\n"
        )
        for i in range(step):
            operation = random.choice(["+", "-"])
            number = random.randint(1, 100)
            if operation == "+":
                final_number += number
                operations += f"{i+1}. Add {number}\n"
            elif operation == "-":
                final_number -= number
                operations += f"{i+1}. Subtract {number}\n"

        return operations, final_number

    def compile_test_entry(self, operations, final_number, operation_step):
        entry_id = self.create_entry_id()
        prompt = self.format_prompt(operations)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": final_number,
            "category": self.task_category,
            "task": self.task_name,
            "step": operation_step,
        }

        return entry

    def compile_task_data(self):
        for step in self.variables["operation_step"]:
            for _ in range(self.num_samples):
                operations, final_number = self.create_context_data(step)
                entry = self.compile_test_entry(operations, final_number, step)
                self.task_data.append(entry)

        return self.task_data


class SetState(Task):
    def __init__(self) -> None:
        super().__init__()

        self.task_name = "set_state"
        self.task_category = "stateful_processing"
        self.num_samples = 5

        self.task_instruction = 'Given the actions of the agents, your task is to determine the final list of words the agent ends up with after a series of actions. Write your final answer after the text "FINAL ANSWER:". For example, "FINAL ANSWER: word1, word2, word3".'

        self.variables = {
            "action_step": [200],
            "state_size": [5, 10, 15, 20],
        }

        self.metrics = ["exact_match", "rouge", "set_overlap"]

    def format_prompt(self, agent_actions):
        prompt = (
            "Agent actions:\n\n"
            + agent_actions
            + "\n\nInstruction:\n"
            + self.task_instruction
            + "\n\nFINAL ANSWER:"
        )

        return prompt

    def create_context_data(self, state_size, step):
        unique_words = random.sample(self.WORDS, state_size * 100)

        agent_actions = ""
        agent_state = []

        actions = ["draw", "discard"]
        for i in range(step):
            action = actions[i % 2]
            if action == "draw":
                if i == 0:
                    n_words = state_size
                else:

                    n_words = random.randint(1, state_size - len(agent_state))
                available_words = [
                    word for word in unique_words if word not in agent_state
                ]
                words = random.sample(available_words, n_words)

                agent_state += words

                agent_actions += (
                    f"Agent draws the following words: {', '.join(words)}.\n"
                )
            elif action == "discard":
                n_words = random.randint(1, int(len(agent_state) / 2))
                words = random.sample(agent_state, n_words)
                agent_state = [word for word in agent_state if word not in words]

                agent_actions += (
                    f"Agent discards the following words: {', '.join(words)}.\n"
                )

        return agent_actions, agent_state

    def compile_test_entry(
        self, agent_actions, agent_final_state, action_step, state_size
    ):
        entry_id = self.create_entry_id()
        prompt = self.format_prompt(agent_actions)
        entry = {
            "id": entry_id,
            "prompt": prompt,
            "reference": agent_final_state,
            "level": self.task_category,
            "task": self.task_name,
            "step": action_step,
            "state_size": state_size,
        }

        return entry

    def compile_task_data(self):
        for step in self.variables["action_step"]:
            for state_size in self.variables["state_size"]:
                for _ in range(self.num_samples):
                    agent_actions, agent_final_state = self.create_context_data(
                        state_size, step
                    )
                    entry = self.compile_test_entry(
                        agent_actions, agent_final_state, step, state_size
                    )
                    self.task_data.append(entry)

        return self.task_data
