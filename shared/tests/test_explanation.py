from shared.medshield.explain.explanation_text import generate_clinical_explanation


def test_generate_clinical_explanation():
    predicted_class = "Tumor"
    confidence_score = 0.946
    heatmap_region = "frontal lobe"

    explanation = generate_clinical_explanation(
        predicted_class=predicted_class,
        confidence_score=confidence_score,
        heatmap_region=heatmap_region,
    )

    expected_explanation = (
        "The model predicts Tumor with 95% confidence, focusing primarily on the frontal lobe area."
    )
    assert explanation == expected_explanation

    # Test rounding and another class
    explanation_2 = generate_clinical_explanation("Healthy", 0.453, "cerebellum")
    assert (
        explanation_2
        == "The model predicts Healthy with 45% confidence, focusing primarily on the cerebellum area."
    )
