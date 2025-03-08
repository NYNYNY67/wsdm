from logzero import logger


def evaluate(df_eval):
    cnt_model_a = df_eval["winner"].value_counts().get("model_a", 0)
    cnt_model_b = df_eval["winner"].value_counts().get("model_b", 0)

    logger.info(f"winner, model_a: {cnt_model_a}, model_b: {cnt_model_b}")

    cnt_model_a = df_eval["pred"].value_counts().get("model_a", 0)
    cnt_model_b = df_eval["pred"].value_counts().get("model_b", 0)

    logger.info(f"prediction, model_a: {cnt_model_a}, model_b: {cnt_model_b}")

    accuracy = (df_eval["winner"] == df_eval["pred"]).mean()
    logger.info(f"accuracy: {accuracy}")
