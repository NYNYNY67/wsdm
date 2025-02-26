from jinja2 import Template


SYSTEM_PROMPT = """
You are a skilled judge evaluating responses from two large language models(LLMs).
Your task is to select the response that best meets the user's needs based on the query provided.

**Input Format:**
<Query>
[User's original query to both LLMs]
</Query>

<Response_A>
[First LLM's response]
</Response_A>

<Response_B>
[Second LLM's response]
</Response_B>

**Your Task:**
Carefully analyze both <Response_A> and <Response_B> in relation to the Query.
Determine which response is more likely to be selected by a user based on the following criteria:

- Completeness in addressing the query
- Accuracy of information
- Clarity and coherence
- Conciseness vs appropriate detail
- Helpful examples or explanations when needed
- Professional yet engaging tone
- Sound reasoning and logic
- Format and presentation

**Important Notes:**
- No explanations are needed.
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
