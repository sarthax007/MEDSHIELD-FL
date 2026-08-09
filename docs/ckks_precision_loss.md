# CKKS Precision Loss Analysis

Homomorphic encryption schemes like CKKS are specifically designed for approximate arithmetic on real numbers. Unlike traditional encryption (e.g. AES) which is strictly bit-exact, CKKS introduces a small, bounded amount of noise into every computation. This noise is what guarantees semantic security.

## How much precision is lost?
In the MedShield-FL framework, our `create_ckks_context` uses a global scale of `2**40`. This reserves 40 bits specifically for the fractional component of our model weights (and updates).

Our automated decryption tests (`test_decryption.py`) mathematically quantify this error across random weight vectors:
- **Mean Absolute Error (MAE)**: Typically on the magnitude of $1 \times 10^{-6}$
- **Max Absolute Error**: Strictly bounded below $1 \times 10^{-3}$

## Impact on Neural Network Accuracy
Deep learning models, and specifically Vision Transformers (ViTs), are inherently robust to small amounts of parameter noise. In fact, injecting noise into weights is sometimes used as a regularization technique to prevent overfitting.

The magnitude of the CKKS decryption noise ($< 10^{-3}$ worst-case) is orders of magnitude smaller than the standard variance in stochastic gradient descent (SGD) updates or the precision lost during quantization (e.g. FP32 $\rightarrow$ FP16 or INT8).

As verified by `test_decryption_preserves_model_predictions`, performing a full round-trip (extract $\rightarrow$ encrypt $\rightarrow$ decrypt $\rightarrow$ load) on the critical parameters alters the final un-softmaxed output logits by less than $1 \times 10^{-3}$. Because classification relies on `argmax` across classes, this microscopic shift in raw logit value does not change the predicted tumor class or meaningfully impact the confidence score. 

**Conclusion:** The mathematical privacy guarantees of CKKS are achieved with a negligible precision trade-off that has **no material impact** on the diagnostic accuracy of MedShield-FL.
