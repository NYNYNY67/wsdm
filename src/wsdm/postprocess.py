def postprocess(df):
    df["pred"] = df["response"].map(
        {
            "A": "model_a",
            "B": "model_b",
        }
    )
    return df
