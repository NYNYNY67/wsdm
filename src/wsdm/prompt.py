from jinja2 import Template


SYSTEM_PROMPT = """
You are a skilled judge evaluating responses from two large language models(LLMs).
Your task is to select the response that best meets the user's needs based on the query provided.

**Important Notes**
- No explanations are needed.
- Just select the response that best meets the user's needs.

**Example Answer**
Choice:
A
"""

USER_PROMPT_TEMPLATE = Template(
"""
Here is your input to process now-

<Query>
{{query}}
</Query>

<Response_A>
{{response_a}}
</Response_A>

<Response_B>
{{response_b}}
</Response_B>

Now, tell me which response you think is better for the user. You can choose either A or B.
"""
)

ASSISTANT_PROMPT = """
Choice:

"""

ASSISTANT_TEMPLATE = Template("""
Choice:
{{answer}}"""
)
