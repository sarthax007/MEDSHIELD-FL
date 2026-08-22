def generate_clinical_explanation(
    predicted_class: str, confidence_score: float, heatmap_region: str
) -> str:
    """
    Generate a plain-language explanation of a model's prediction suitable for a clinician.

    Args:
        predicted_class (str): The class predicted by the model (e.g., "Tumor", "Normal").
        confidence_score (float): The confidence score of the prediction, as a decimal between 0 and 1.
        heatmap_region (str): A description of the anatomical or spatial region highlighted by the CAM.

    Returns:
        str: A short, non-technical sentence explaining the prediction.
    """
    confidence_percentage = int(round(confidence_score * 100))

    return (
        f"The model predicts {predicted_class} with {confidence_percentage}% confidence, "
        f"focusing primarily on the {heatmap_region} area."
    )
