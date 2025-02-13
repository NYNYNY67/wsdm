import pandas as pd
from transformers import AutoTokenizer

from wsdm.prompt import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    ASSISTANT_PROMPT,
)


def preprocess(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    system_prompt: str = SYSTEM_PROMPT,
    assistant_prompt: str = ASSISTANT_PROMPT,
):
    """
    Preprocess the data.

    Args:
        df: The DataFrame to preprocess.
        is_train: Whether the DataFrame is the training data.

    Returns:
        The preprocessed DataFrame.
    """
    df = df.copy()
    df = render_user_prompt(df)
    df = apply_chat_template(df, tokenizer, system_prompt, assistant_prompt)

    return df


def render_user_prompt(
    df: pd.DataFrame,
):
    """
    Render the user prompt.

    Args:
        df: The DataFrame to render the user prompt for.

    Returns:
        The DataFrame with the user prompt rendered.
    """
    df = df.copy()

    df["user_prompt"] = df.apply(
        lambda x: USER_PROMPT_TEMPLATE.render(
            query=x["prompt"],
            response_a=x["response_a"],
            response_b=x["response_b"],
        ),
        axis=1,
    )

    return df


def apply_chat_template(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    system_prompt: str = SYSTEM_PROMPT,
    assistant_prompt: str = ASSISTANT_PROMPT,
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

    df["text"] = df.apply(
        lambda x: [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": x["user_prompt"]},
            {"role": "assistant", "content": assistant_prompt},
        ],
        axis=1,
    )
    df["text"] = df["text"].apply(
        lambda x: tokenizer.apply_chat_template(
            x,
            tokenize=False,
            add_general_prompt=False,
            continue_final_message=True,
        ),
    )

    return df
