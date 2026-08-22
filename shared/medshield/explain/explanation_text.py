import numpy as np


def determine_heatmap_region(cam_mask: np.ndarray) -> str:
    """
    Determine the anatomical or spatial region highlighted by the CAM.
    Calculates the highest intensity point from the CAM mask and maps it to a string.
    """
    if cam_mask.size == 0:
        return "unknown"

    # Get coordinates of maximum intensity
    y, x = np.unravel_index(np.argmax(cam_mask), cam_mask.shape)
    h, w = cam_mask.shape

    # Define center bounding box (middle third)
    if h / 3 <= y < 2 * h / 3 and w / 3 <= x < 2 * w / 3:
        return "center"

    if y < h / 2:
        if x < w / 2:
            return "upper-left quadrant"
        else:
            return "upper-right quadrant"
    else:
        if x < w / 2:
            return "lower-left quadrant"
        else:
            return "lower-right quadrant"


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
    confidence_percentage = round(confidence_score * 100)

    return (
        f"The model predicts {predicted_class} with {confidence_percentage}% confidence, "
        f"focusing primarily on the {heatmap_region} area."
    )
