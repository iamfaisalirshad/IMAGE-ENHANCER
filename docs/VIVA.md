# IMAGE ENHANCER — Viva Questions & Answers

20+ questions covering CNN, restoration, super-resolution, metrics, training,
and hardware for the IMAGE ENHANCER project.

---

### 1. What is a CNN?
A Convolutional Neural Network is a deep model that uses convolutional layers
to automatically learn spatial features (edges, textures, patterns) from
images, using learned filters/kernels.

### 2. What is convolution?
An operation where a small filter (kernel) slides across an input and computes
element-wise dot products to produce a feature map, detecting local patterns.
Example: 3x3 filter with stride 1 and padding 1 keeps spatial size.

### 3. What is feature extraction?
The process of learning meaningful representations (like edges and textures)
from raw pixels through successive convolutional layers. Each layer's output
is a feature map.

### 4. What is image restoration?
Recovering a clean image from a degraded version (noisy, blurred, or
compressed). In this project we also restore spatial detail.

### 5. What is image enhancement / super-resolution?
Super-resolution reconstructs a higher-resolution image from a lower-resolution
one. Image enhancement improves visual quality/perceptibility.

### 6. What is the difference between restoration and enhancement?
Restoration tries to recover the original signal using a mathematical model of
degradation; enhancement improves appearance for viewing without necessarily
reverting a known degradation.

### 7. What are residual blocks?
Blocks where the input is added back to the output of a few layers
(`y = F(x) + x`). This skip connection eases training of deep networks and
avoids vanishing gradients, letting the network learn only the "residual".

### 8. Why use a skip connection (global skip in V2)?
It lets strong structural information flow directly to the output and makes
learning easier. In V2 the bicubic-upscaled input is added to the output.

### 9. What is PixelShuffle?
An operation that rearranges channels into spatial size, e.g. turning an
`r^2 * C` channel map into `C` channels at `r` times the spatial size. Used for
learned 2x upsampling. Here `r = 4` channels -> 1 channel at 2x size.

### 10. What is L1 loss (MAE)?
Mean Absolute Error between prediction and target. It is robust to outliers and
often used in restoration because it produces less blur than L2/MSE.

### 11. What is PSNR and why is it used?
Peak Signal-to-Noise Ratio (in dB) = `10*log10(MAX^2/MSE)`. It measures the
ratio between the maximum possible power and the corrupting noise power.
Higher is better. It does not always match human perception, so SSIM is added.

### 12. What is SSIM?
Structural Similarity Index Mean (0–1). It compares luminance, contrast, and
structure between two images. Values closer to 1 mean more similar.

### 13. What is bicubic interpolation?
A traditional upscaling method that estimates new pixels using a cubic
polynomial of surrounding pixels. It is our baseline, which the CNNs must beat,
since it cannot restore true missing detail.

### 14. What is ground truth?
The clean, original (high-resolution) image used as the target during training
and as the reference when computing PSNR/SSIM.

### 15. What is the difference between training and validation?
During training the model updates weights to fit the training data. Validation
measures how well the model generalizes to unseen data (and helps detect
overfitting).

### 16. What is an epoch?
One complete pass over the entire training dataset. More epochs usually mean
lower training loss, but may overfit.

### 17. What is batch size?
The number of training samples processed before the model weights are updated
once. It affects memory and update frequency.

### 18. What is learning rate?
The step size for updating weights during optimization. Too high -> unstable;
too low -> slow convergence.

### 19. What is CUDA?
NVIDIA's parallel computing platform that lets PyTorch run matrix operations on
the GPU, making training much faster than CPU.

### 20. What is CPU vs GPU for this project?
CPU runs every core but is slower for large matrix ops; GPU (CUDA) runs
thousands of threads in parallel, ideal for CNNs. The project auto-detects GPU
and falls back to CPU so it works anywhere.

### 21. What is overfitting?
When the model memorizes the training data but fails on new data (high training
accuracy, poor validation). Mitigated by validation, more data, and early
stopping.

### 22. What is the input and output shape of your models?
Input `[B, 3, 128, 128]` (RGB low-res), output `[B, 3, 256, 256]` (RGB
high-res), where B is the batch size.

### 23. How did you handle RGB images in SSIM?
We kept the channel dimension and used `skimage.metrics.structural_similarity`
with `channel_axis` set, averaging across channels.

### 24. Why is bicubic not enough for super-resolution?
Bicubic only interpolates pixels; it cannot invent missing high-frequency
detail and tends to blur edges. A trained CNN learns the mapping to sharp
details.

### 25. What are your project's limitations?
Quick Demo gives low quality; full quality needs a real dataset and more
training time. Results depend on hardware and dataset size.

### 26. How do you know which model is better?
By comparing average PSNR (higher) and SSIM (higher) on the validation set for
Bicubic, V1, and V2 using the same images.

### 27. Why save only the best validation checkpoint?
To keep only the model that generalizes best on unseen data, avoiding storing
many large checkpoint files and providing automatic best-model selection.
