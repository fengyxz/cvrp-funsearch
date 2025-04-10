class PromptGenerator:
    def __init__(
        self, user_prompt, code_template, avoid_errors, data=None
    ):
        self.user_prompt = user_prompt
        self.code_template = code_template
        self.avoid_errors = avoid_errors
        self.data = data

    def generate(self, data=None):
        if data:
            self.data = data
        avoid_errors_str = "\n Please avoid these errorn\n" + "\n".join(
            self.avoid_errors
        )
        prompt = f"{self.user_prompt}{self.code_template}\n data: {self.data}\n{avoid_errors_str} "
        return prompt
