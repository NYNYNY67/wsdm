import pandas as pd
from transformers import AutoTokenizer

from wsdm.prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    ASSISTANT_PROMPT,
    ASSISTANT_TEMPLATE,
)


def render_templates(
    df: pd.DataFrame,
    with_answer: bool,
    response_max_length: int,
):
    """
    Render the user prompt.

    Args:
        df: The DataFrame to render the user prompt for.

    Returns:
        The DataFrame with the user prompt rendered.
    """
    df = df.copy()

    df["system_prompt"] = SYSTEM_PROMPT
    df["user_prompt"] = df.apply(
        lambda x: USER_PROMPT_TEMPLATE.render(
            query=x["prompt"],
            response_a=x["response_a"][:response_max_length],
            response_b=x["response_b"][:response_max_length],
        ),
        axis=1,
    )

    if with_answer:
        df["assistant_prompt"] = df.apply(
            lambda x: ASSISTANT_TEMPLATE.render(answer=x["winner"][-1].upper()),
            axis=1,
        )
    else:
        df["assistant_prompt"] = ASSISTANT_PROMPT

    return df


def get_chat_conversation(
    df: pd.DataFrame,
):
    """
    Get the chat conversation from the DataFrame.

    Args:
        df: The DataFrame to get the chat conversation from.
        system_prompt: The system prompt to use.
        assistant_prompt: The assistant prompt to use.

    Returns:
        The DataFrame with the chat conversation.
    """
    df["text"] = df.apply(
        lambda x: [
            {"role": "system", "content": x["system_prompt"]},
            {"role": "user", "content": x["user_prompt"]},
            {"role": "assistant", "content": x["assistant_prompt"]},
        ],
        axis=1,
    )
    return df


def apply_chat_template(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
):
    """
    Apply the chat template to the DataFrame.

    Args:
        df: The DataFrame to apply the chat template to.
        tokenizer: The tokenizer to use.
        system_prompt: The system prompt to use.
        assistant_prompt: The assistant prompt to use.

    Returns:
        The DataFrame with the chat template applied.
    """
    df = df.copy()
    df["text"] = df["text"].apply(
        lambda x: tokenizer.apply_chat_template(
            x,
            tokenize=False,
            add_general_prompt=False,
            continue_final_message=True,
        ),
    )

    return df
