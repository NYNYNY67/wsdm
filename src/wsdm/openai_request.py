import os
import pandas as pd
from openai import OpenAI


def get_completion(
    df: pd.DataFrame,
    model: str,
):
    """
    Get the completion from the DataFrame.

    Args:
        df: The DataFrame to get the completion from.
        model: The model to use.
    """

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for index, row in df.iterrows():
        response = client.chat.completions.create(
            messages=row["text"],
            model=model,
        )
        df.loc[index, "response"] = response.choices[0].message.content[-1]

    return df
