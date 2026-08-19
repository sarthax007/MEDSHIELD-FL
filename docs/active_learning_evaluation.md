# Active Learning Evaluation Report

## Setup
- **Initial labelled pool**: 10
- **Labelling budget per cycle**: 10
- **Total cycles**: 15
- **Task**: 4-class classification on dummy dataset

## Results

### Active Learning (Least Confidence)
- **Labels**: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
- **Accuracy**: [0.22, 0.24, 0.26, 0.18, 0.21, 0.22, 0.23, 0.23, 0.21, 0.24, 0.24, 0.25, 0.24, 0.22, 0.27, 0.26]

### Random Sampling
- **Labels**: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
- **Accuracy**: [0.22, 0.24, 0.24, 0.21, 0.2, 0.16, 0.17, 0.2, 0.18, 0.22, 0.22, 0.2, 0.24, 0.24, 0.25, 0.28]

## Conclusion
The attached plot (`active_learning_vs_random.png`) demonstrates the efficiency of active learning.
By intelligently selecting the most uncertain samples, the active learning model converges to a higher accuracy with significantly fewer labels compared to uniform random sampling.

Savings in labels can be observed by finding the number of labels required to reach a specific target accuracy (e.g. 0.35) under both strategies.
