"""
IMAGE ENHANCER - Central Configuration
=======================================
All the settings for the project live here.
You can tweak values without touching any other file.
"""

import os

# ---------------------------------------------------------------------------
# Paths (auto-created on startup)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
COMPARISON_DIR = os.path.join(RESULTS_DIR, "comparisons")
ENHANCED_DIR = os.path.join(RESULTS_DIR, "enhanced")
CURVE_DIR = os.path.join(RESULTS_DIR, "training_curves")
METRIC_DIR = os.path.join(RESULTS_DIR, "metrics")

# Where trained weights are saved / loaded
CHECKPOINT_V1 = os.path.join(CHECKPOINT_DIR, "best_model_v1.pth")
CHECKPOINT_V2 = os.path.join(CHECKPOINT_DIR, "best_model_v2.pth")

# ---------------------------------------------------------------------------
# Image sizes
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128          # input (low-res)
TARGET_SIZE = 256         # output (hi-res)

# ---------------------------------------------------------------------------
# Training defaults (used when the user does not change them)
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_WORKERS = 0           # 0 is safest for laptops/Windows
SEED = 42

# How many images per split to load. None = use all available.
DATASET_LIMIT = None      # e.g. 50 to cap the dataset

# ---------------------------------------------------------------------------
# Degradation pipeline
# ---------------------------------------------------------------------------
BLUR_SIGMA = 1.2          # Gaussian blur kernel sigma (fixed default)
NOISE_LEVEL = 0.03        # Gaussian noise standard deviation (0-1 scale)
JPEG_QUALITY = 85         # optional JPEG compression quality
DO_JPEG = False           # enable/disable JPEG compression
BRIGHTNESS_FACTOR = 1.0   # 1.0 = unchanged
CONTRAST_FACTOR = 1.0     # 1.0 = unchanged

# Random ranges used during training to make the model robust
# (the degradation is strong enough that the CNN can clearly beat bicubic)
BLUR_SIGMA_MIN = 0.3
BLUR_SIGMA_MAX = 2.2
NOISE_MIN = 0.0
NOISE_MAX = 0.05

# ---------------------------------------------------------------------------
# Quick demo training settings
# ---------------------------------------------------------------------------
QUICK_NUM_IMAGES = 12     # images used in quick mode (max)
QUICK_EPOCHS = 3

# Number of synthetic sample images generated on first run (if empty)
SAMPLE_COUNT = 40

# ---------------------------------------------------------------------------
# Enhancement output sharpening
# ---------------------------------------------------------------------------
# A mild unsharp-mask pass is applied to the enhanced output so the result
# is visibly crisp without overprocessing. Metrics (PSNR/SSIM vs bicubic)
# are computed WITHOUT this pass, so the honest numbers are unaffected.
ENHANCE_SHARPEN_AMOUNT = 1.4   # 0 = off, 1.0 = default strength
SHARPEN_RADIUS = 2.0
SHARPEN_THRESHOLD = 2

# How much to soften each input tile before the model sees it (float sigma,
# in pixels). 0 = feed the tiles exactly as they are. Ablation showed that
# softening already-clean tiles DESTROYS native detail (sharper input is
# better for visibly enhanced output), so this stays 0 by default.
TILE_PRE_BLUR = 0.0

# ---------------------------------------------------------------------------
# Adaptive enhancement ("restore only what is actually wrong")
# ---------------------------------------------------------------------------
# The input image is measured before enhancement. A "deficiency" score in
# [0,1] says how blurred/noisy it really is. Clean/sharp images keep the
# original almost untouched (floor alpha, no sharpening), while genuinely
# blurred or noisy images get the full model restoration + stronger
# sharpening. Per-channel tone means are always matched back to the input so
# shades/colors are never shifted. Set ENHANCE_ADAPTIVE = False to use the
# fixed full-strength pipeline instead.
ENHANCE_ADAPTIVE = True
ADAPTIVE_KEEP_IF_CLEAN = 0.08   # deficiency below this = already clean: keep original 2x
ADAPTIVE_MIN_ALPHA = 0.25       # minimum share of the model result kept (rest goes to input)
ADAPTIVE_MAX_ALPHA = 1.0        # ceiling for the model share (hit exactly at peak deficiency)
ADAPTIVE_ALPHA_SLOPE = 1.00     # alpha = min(max, min_alpha + slope * deficiency)

# Deficiency measurement (on a 256-normalised grayscale version of the input):
# blur is judged by how fast edge energy collapses under extra smoothing, as
# a ratio that is independent of content and robust to noise. Two probes are
# used and a blur is flagged if EITHER sees it.
BLUR_RATIO_LO = 1.6     # probe 1 ratio <= this = badly blurred (restore fully)
BLUR_RATIO_HI = 2.6     # probe 1 ratio >= this = already sharp (do nothing)
BLUR_RATIO2_HI = 6.0    # probe 2 (noise-robust) ratio >= this = sharp
NOISE_STD_LO = 2.5      # robust noise est. (0-255 units) <= this = clean
NOISE_STD_HI = 8.0      # noise est. >= this = very noisy

# ---------------------------------------------------------------------------
# Adaptive shade ("veil / washed-out") correction
# ---------------------------------------------------------------------------
# Some degraded photos look flat and hazy: their dynamic range is compressed
# and the black point is lifted, which reads as an unwanted grey "shade" over
# the picture. If that is really measured the tones are corrected with a
# plain de-veiling step (contrast gain pivoted at the median luminance plus a
# small saturation lift). Shadows are NEVER touched: the gain is always >= 1,
# so dark/shadow pixels only get deeper, never lifted, and healthy photos
# with real shadows measure a wide dynamic range and skip this entirely.
SHADE_RESTORE_KEEP_IF_OK = 0.12  # measured shade below this -> leave the tones alone
SHADE_GAIN_MAX = 0.28            # maximum extra contrast gain applied (x1.00 .. x1.28)
SHADE_SAT_MAX = 0.15             # maximum extra saturation gain when veiled
SHADE_RANGE_LO = 0.32            # luminance pass-band (p98-p2) below this = severely veiled
SHADE_RANGE_HI = 0.52            # pass-band above this = healthy range, no shade correction
SHADE_BLACK_LIFT_LO = 0.065      # 5th-%tile luminance above this starts to signal lifted blacks
SHADE_BLACK_LIFT_HI = 0.16       # 5th-%tile luminance at/above this = clearly lifted blacks

# ---------------------------------------------------------------------------
# Graphics
# ---------------------------------------------------------------------------
DEVICE_CPU_LABEL = "CPU"
DEVICE_CUDA_LABEL = "NVIDIA CUDA GPU"

# ---------------------------------------------------------------------------
# List of all folders that MUST exist.
# ---------------------------------------------------------------------------
REQUIRED_DIRS = [
    TRAIN_DIR,
    VAL_DIR,
    SAMPLE_DIR,
    CHECKPOINT_DIR,
    COMPARISON_DIR,
    ENHANCED_DIR,
    CURVE_DIR,
    METRIC_DIR,
]
