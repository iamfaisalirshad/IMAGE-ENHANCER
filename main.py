"""
IMAGE ENHANCER
==============
AI-Based Image Restoration and 2x Super-Resolution Using CNN.

PRIMARY ENTRY POINT:
    python main.py

Run the application, check the environment on first run, and
present a friendly menu / GUI so you never touch the internals.
"""

import os
import sys

import config

# ---------------------------------------------------------------------------
# FIRST-RUN: make sure folders exist before anything else.
# ---------------------------------------------------------------------------
from utils.file_utils import ensure_dirs

ensure_dirs(config.REQUIRED_DIRS)

PROJECT_NAME = "IMAGE ENHANCER"
TECH_TITLE = "AI-Based Image Restoration and 2x Super-Resolution Using CNN"


def seed_sample_images():
    """If data/sample/ is empty, generate small synthetic sample images
    so Quick Demo training and comparisons work without a dataset."""
    from utils.file_utils import find_images
    if len(find_images(config.SAMPLE_DIR)) == 0:
        try:
            from utils.sample_gen import generate_samples
            n = generate_samples()
            print(f"[INIT] Generated {n} synthetic sample images in "
                  f"data/sample/ for demo purposes.")
        except Exception as e:
            print(f"[INIT] Could not auto-generate sample images: {e}")



def environment_summary():
    """Return a dict describing the local environment (no heavy imports)."""
    summary = {"folders_ok": True}
    try:
        from utils.device import device_summary
        summary["device"] = device_summary()
    except Exception:
        summary["device"] = "unknown"

    # Trained model availability
    summary["v1"] = os.path.isfile(config.CHECKPOINT_V1)
    summary["v2"] = os.path.isfile(config.CHECKPOINT_V2)

    # Dataset availability
    from utils.file_utils import find_images
    summary["train_images"] = len(find_images(config.TRAIN_DIR))
    summary["val_images"] = len(find_images(config.VAL_DIR))
    summary["sample_images"] = len(find_images(config.SAMPLE_DIR))
    return summary


def welcome_banner(env):
    box = "=" * 52
    lines = [
        box,
        f"                {PROJECT_NAME}",
        f"       {TECH_TITLE}",
        box,
        f"  Device     : {env['device']}",
        f"  Train imgs : {env['train_images']}   Val imgs : {env['val_images']}",
        f"  V1 model   : {'trained' if env['v1'] else 'NOT trained'}",
        f"  V2 model   : {'trained' if env['v2'] else 'NOT trained'}",
        box,
    ]
    return "\n".join(lines)


# ===========================================================================
# TERMINAL INTERFACE
# ===========================================================================
def safe_input(prompt=""):
    """
    Wrapper around built-in input() that never crashes the program.
    Raises SystemExit to exit cleanly if stdin is closed (EOF) or the
    user presses Ctrl+C — so main.py never shows an ugly traceback
    when run non-interactively or from an IDE.
    """
    try:
        return input(prompt)
    except EOFError:
        print("\n[Exiting] Input stream closed. Goodbye!")
        raise SystemExit(0)
    except KeyboardInterrupt:
        print("\n[Exiting] Interrupted. Goodbye!")
        raise SystemExit(0)


def pause():
    """Wait for the user to press Enter (never crashes on EOF)."""
    try:
        safe_input("\nPress Enter to continue...")
    except SystemExit:
        pass


def run_terminal():
    seed_sample_images()
    env = environment_summary()

    # Friendly first-run notices (non-fatal).
    if env["train_images"] == 0 or env["val_images"] == 0:
        print("\n[INFO] Dataset not found or empty.")
        print("       Please place your training images inside "
              "data/train/ and data/val/.")
        if env["sample_images"] == 0:
            print("       (Tip: put a few sample images in data/sample/ too.)")
        print("       You can still try 'Quick Demo' training using "
              "sample images.")
    if not env["v1"] and not env["v2"]:
        print("\n[INFO] No trained models found. Use menu option 2 "
              "(Train Model) to train V1/V2 first.")

    while True:
        print()
        print(welcome_banner(env))
        print("\n 1. Enhance an Image")
        print(" 2. Train Model")
        print(" 3. Evaluate Models")
        print(" 4. Compare V1 vs V2")
        print(" 5. View Saved Results")
        print(" 6. Launch GUI")
        print(" 7. Project Information")
        print(" 8. Exit")
        choice = safe_input("\nEnter your choice: ").strip()

        if choice == "1":
            menu_enhance()
        elif choice == "2":
            menu_train(env)
        elif choice == "3":
            menu_evaluate(env)
        elif choice == "4":
            menu_compare(env)
        elif choice == "5":
            menu_results()
        elif choice == "6":
            run_gui()
        elif choice == "7":
            menu_about()
        elif choice == "8":
            print("\nGoodbye!")
            break
        else:
            print("\n[Error] Please enter a valid option (1-8).")

        # refresh env after actions (e.g. training may add models)
        env = environment_summary()


def _pick_image(prompt="Enter image path: "):
    """Ask for an image path and return it, or None on failure."""
    p = safe_input(prompt).strip().strip('"').strip("'")
    if not p:
        print("[Info] No path entered. Cancelled.")
        return None
    if not os.path.isfile(p):
        print(f"[Error] Could not find file: {p}")
        return None
    return p


def _pick_model():
    choice = safe_input("Select model [1=V1, 2=V2, default=2]: ").strip()
    if choice == "1":
        return "v1"
    return "v2"


def menu_enhance():
    env = environment_summary()
    name = _pick_model()
    if not env[name]:
        print(f"\n[Error] No trained {name.upper()} model found. "
              f"Please train {name.upper()} first (menu 2).")
        return

    path = _pick_image("Enter path to image (128x128 or larger): ")
    if not path:
        return

    try:
        from inference.predictor import load_model, enhance_file
        from utils.device import get_device
        from utils.file_utils import save_result_image
        import numpy as np

        dev = get_device()
        model = load_model(name, dev)
        low, enhanced = enhance_file(model, path)
        saved = save_result_image(enhanced, config.ENHANCED_DIR,
                                  prefix=f"enhanced_{name.upper()}")
        print(f"\n[Done] Enhanced using {name.upper()}.")
        print(f"       Output size: 2x the original (true super-resolution).")
        print(f"       Output saved to: {saved}")
    except Exception as e:  # friendly error
        print(f"\n[Error] Could not complete enhancement: {e}")
        print("        Please check the image file is valid.")


def _train_setup_kind():
    print("\n Training mode:")
    print("  1. Quick Demo (few images, low epochs — verify it works)")
    print("  2. Normal")
    kind = safe_input("Choose [1/2, default=2]: ").strip()
    return kind


def menu_train(env):
    print("\n Which model?")
    print("  1. Train V1")
    print("  2. Train V2")
    print("  3. Train Both")
    which = safe_input("Choose [1/2/3]: ").strip()
    if which not in ("1", "2", "3"):
        print("\n[Error] Please enter a valid option (1, 2 or 3).")
        return

    kind = _train_setup_kind()
    quick = kind == "1"

    model_names = {"1": ["v1"], "2": ["v2"], "3": ["v1", "v2"]}[which]

    if quick:
        epochs_in = safe_input(f"Epochs [default={config.QUICK_EPOCHS}]: ").strip()
        epochs = int(epochs_in) if epochs_in.isdigit() else config.QUICK_EPOCHS
        limit = config.QUICK_NUM_IMAGES
        print(f"\n[Quick Demo] Using up to {limit} images per split, "
              f"{epochs} epoch(s).")
        print("NOTE: Quick Demo gives low quality — it only verifies the "
              "pipeline works.")
        run_training(model_names, epochs, quick=True, limit=limit)
        return

    print("\n[Normal Training]")
    print("Training time depends on your CPU/GPU, dataset size, batch size, "
          "and epochs.")
    epochs_in = safe_input(f"Epochs [default={config.EPOCHS}]: ").strip()
    epochs = int(epochs_in) if epochs_in.isdigit() else config.EPOCHS
    bs_in = safe_input(f"Batch size [default={config.BATCH_SIZE}]: ").strip()
    batch_size = int(bs_in) if bs_in.isdigit() else config.BATCH_SIZE
    lr_in = safe_input(f"Learning rate [default={config.LEARNING_RATE}]: ").strip()
    try:
        lr = float(lr_in) if lr_in else config.LEARNING_RATE
    except ValueError:
        lr = config.LEARNING_RATE

    print("\n Dataset size for training (images per split):")
    print("  0 = use all available images")
    ds_in = safe_input("Limit [default=0]: ").strip()
    limit = int(ds_in) if ds_in.isdigit() and int(ds_in) > 0 else None

    run_training(model_names, epochs, quick=False, batch_size=batch_size,
                 lr=lr, limit=limit)


def run_training(model_names, epochs, quick=False, batch_size=None, lr=None,
                 limit=None):
    try:
        from utils.seed import set_seed
        from utils.device import get_device
        from dataset.image_dataset import make_dataloaders
        from training.trainer import train_model
        from models.cnn_v1 import ImageEnhancerV1
        from models.cnn_v2 import ImageEnhancerV2
        from visualization.visualizer import plot_training_history
        import config as cfg

        set_seed(cfg.SEED)
        device = get_device()
        print(f"\n[Using device: {device}]")

        # Quick mode uses sample images even if train/val dirs are empty.
        train_dir = cfg.TRAIN_DIR
        val_dir = cfg.VAL_DIR
        if quick:
            train_dir = cfg.SAMPLE_DIR
            val_dir = cfg.SAMPLE_DIR
            if limit is None:
                limit = cfg.QUICK_NUM_IMAGES

        train_loader, val_loader, summary = make_dataloaders(
            train_dir, val_dir, batch_size=batch_size or cfg.BATCH_SIZE,
            limit=limit,
        )
        print(f"\n[TRAIN] {summary['train_images']} imgs, "
              f"[VAL] {summary['val_images']} imgs.")

        if summary["train_images"] == 0 or summary["val_images"] == 0:
            print("\n[Error] No images available for training.")
            print("        Add images to data/train/ and data/val/, "
                  "or use Quick Demo with data/sample/.")
            return

        factories = {
            "v1": (ImageEnhancerV1, cfg.CHECKPOINT_V1),
            "v2": (ImageEnhancerV2, cfg.CHECKPOINT_V2),
        }
        histories = []
        lr = lr or cfg.LEARNING_RATE

        for name in model_names:
            cls, ckpt = factories[name]
            print(f"\n=== Training {name.upper()} ===")
            model = cls()
            kwargs = {}
            if quick:
                kwargs["quit_flag"] = {"quit": False}
            hist = train_model(
                model, name, train_loader, val_loader, device,
                epochs=epochs, learning_rate=lr, checkpoint_path=ckpt,
                seed=cfg.SEED,
            )
            histories.append((name, hist))

        print("\n[TRAINING COMPLETE]")
        for name, hist in histories:
            last = hist["val_psnr"][-1] if hist["val_psnr"] else 0
            print(f"  {name.upper()}: final val PSNR ~ {last:.2f} dB")

        if histories:
            try:
                saved = plot_training_history(histories)
                print("\n  Training curves saved under "
                      "results/training_curves/")
            except Exception as e:
                print(f"\n  (Could not plot curves: {e})")
    except Exception as e:
        print(f"\n[Error] Training failed: {e}")


def menu_evaluate(env):
    if not env["v1"] and not env["v2"]:
        print("\n[Error] No trained models. Train a model first (menu 2).")
        return
    try:
        from utils.device import get_device
        from dataset.image_dataset import make_dataloaders
        from inference.predictor import load_model
        from evaluation.evaluator import evaluate_model
        from visualization.visualizer import plot_metric_bars
        import config as cfg

        device = get_device()
        from utils.file_utils import find_images

        # Pick a source that has images (val first, then sample).
        source = cfg.VAL_DIR
        if len(find_images(cfg.VAL_DIR)) == 0:
            source = cfg.SAMPLE_DIR
        if len(find_images(source)) == 0:
            print("\n[Error] No validation images. Add some to data/val/ "
                  "or data/sample/.")
            return

        train_loader, val_loader, summary = make_dataloaders(
            source, source, batch_size=cfg.BATCH_SIZE,
            limit=cfg.QUICK_NUM_IMAGES,
        )

        results = []
        for name in ("v1", "v2"):
            if not env[name]:
                print(f"\n[Skipped] {name.upper()} not trained.")
                continue
            model = load_model(name, device)
            print(f"\n=== Evaluating {name.upper()} vs Bicubic ===")
            res = evaluate_model(model, val_loader, device, name)
            print(f"  Bicubic : PSNR {res['avg_psnr_bicubic']:.2f} dB, "
                  f"SSIM {res['avg_ssim_bicubic']:.4f}")
            print(f"  {name.upper()} : PSNR {res['avg_psnr_model']:.2f} dB, "
                  f"SSIM {res['avg_ssim_model']:.4f}")
            results.append(res)

        if results:
            try:
                plot_metric_bars(results, metric="psnr")
                plot_metric_bars(results, metric="ssim")
                print("\n  Metric charts saved under results/training_curves/")
            except Exception as e:
                print(f"\n  (Could not plot metrics: {e})")
    except Exception as e:
        print(f"\n[Error] Evaluation failed: {e}")


def menu_compare(env):
    if not env["v1"] or not env["v2"]:
        print("\n[Error] Both V1 and V2 must be trained to compare. "
              "Train them in menu 2.")
        return
    try:
        import torch
        from utils.device import get_device
        from inference.predictor import load_model
        from utils.file_utils import find_images
        from evaluation.metrics import bicubic_upscale, psnr, ssim
        from visualization.visualizer import save_comparison_grid
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        import config as cfg

        device = get_device()
        images = find_images(cfg.SAMPLE_DIR)
        if not images:
            images = find_images(cfg.VAL_DIR)
        if not images:
            print("\n[Error] No sample/val images for comparison.")
            return

        v1 = load_model("v1", device)
        v2 = load_model("v2", device)
        to_tensor = transforms.ToTensor()

        made = []
        for path in images[:min(2, len(images))]:
            pil = Image.open(path).convert("RGB")
            pil = pil.resize((cfg.TARGET_SIZE, cfg.TARGET_SIZE), Image.BICUBIC)
            target = np.array(pil)
            low_t = to_tensor(pil.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                                         Image.BICUBIC)).unsqueeze(0) * 1.0
            with torch.no_grad():
                p1 = v1.infer_tensor(low_t)
                p2 = v2.infer_tensor(low_t)
            bic = bicubic_upscale(low_t, cfg.TARGET_SIZE)

            def arr(t):
                return (t[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255)\
                    .astype(np.uint8)

            p1a, p2a, bica = arr(p1), arr(p2), arr(bic)
            low_resized = np.array(pil.resize((cfg.TARGET_SIZE, cfg.TARGET_SIZE),
                                              Image.BICUBIC))

            t_tensor = to_tensor(pil).unsqueeze(0)
            p1_ps = psnr(p1, t_tensor)
            p2_ps = psnr(p2, t_tensor)
            b_ps = psnr(bic, t_tensor)

            name = f"comparison_{os.path.splitext(os.path.basename(path))[0]}"
            saved = save_comparison_grid(
                low_resized, low_resized, bica, p1a, target,
                name=name,
                titles=["Degraded Input", "Bicubic",
                        f"V1 (PSNR {p1_ps:.1f})", "Ground Truth"],
            )
            made.append((name, saved, b_ps, p1_ps, p2_ps))

        print("\n[COMPARISON] Saved to results/comparisons/")
        for n, s, b, p1, p2 in made:
            print(f"  {n}: Bicubic {b:.1f} | V1 {p1:.1f} | V2 {p2:.1f} dB")
    except Exception as e:
        print(f"\n[Error] Comparison failed: {e}")


def menu_results():
    from utils.file_utils import find_images
    results_dir = config.RESULTS_DIR

    def list_dir(d, label):
        files = find_images(d)
        sub = os.listdir(d) if os.path.isdir(d) else []
        print(f"\n  {label}:")
        for f in files[:8]:
            print(f"    {os.path.basename(f)}")
        if not files:
            print("    (none)")

    print("\n=== SAVED RESULTS ===")
    list_dir(config.ENHANCED_DIR, "Enhanced images")
    list_dir(config.COMPARISON_DIR, "Comparisons")
    list_dir(config.CURVE_DIR, "Training curves / charts")

    # JSON metrics summary from training
    metrics_files = [f for f in os.listdir(config.METRIC_DIR)]
    if metrics_files:
        import json
        print("\n  Saved metric files:")
        for f in metrics_files:
            p = os.path.join(config.METRIC_DIR, f)
            try:
                with open(p) as fh:
                    data = json.load(fh)
                print(f"    {f}: PSNR {data.get('avg_psnr_model', 0):.2f} dB, "
                      f"SSIM {data.get('avg_ssim_model', 0):.4f}")
            except Exception:
                print(f"    {f}: (unreadable)")

    if not (find_images(config.ENHANCED_DIR) or find_images(config.COMPARISON_DIR)
            or find_images(config.CURVE_DIR) or metrics_files):
        print("\n[Info] No evaluation results found. Train and evaluate a "
              "model first.")

    pause()


def menu_about():
    lines = [
        "\n==================================================",
        f"  {PROJECT_NAME}",
        f"  {TECH_TITLE}",
        "==================================================",
        "  Technologies: Python, PyTorch, CNN, OpenCV,",
        "                 Pillow, NumPy, scikit-image, Matplotlib",
        "",
        "  Purpose: Restore degraded images and generate 2x",
        "           higher-resolution enhanced images.",
        "",
        "  Models: V1 (baseline CNN), V2 (residual + PixelShuffle)",
        "  Input:  128x128 RGB",
        "  Output: 256x256 RGB",
        "==================================================",
    ]
    print("\n".join(lines))
    pause()


# ===========================================================================
# GUI (Tkinter) — reliable local interface
# ===========================================================================
def run_gui():
    """Launch a lightweight Tkinter GUI (falls back to nothing on failure)."""
    try:
        from interface import gui
        gui.launch()
    except Exception as e:
        print(f"\n[Error] Could not launch GUI (check Tkinter): {e}")


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def main():
    print()
    print("=" * 52)
    print(f"   {PROJECT_NAME}")
    print(f"   {TECH_TITLE}")
    print("=" * 52)

    # Decide interface: GUI if explicitly desired, else terminal menu.
    arg = [a for a in sys.argv[1:] if a.lower() in ("--gui", "-g", "--terminal", "-t")]
    if arg and arg[0].lower() in ("--gui", "-g"):
        run_gui()
        return
    run_terminal()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # graceful exit from safe_input / pause
        pass
    except Exception as e:
        print("\n[Error] An unexpected problem occurred: "
              f"{type(e).__name__}: {e}")
        print("        If this keeps happening, check:")
        print("        - dependencies are installed: "
              "pip install -r requirements.txt")
        print("        - you are using the project's virtual environment")
        print("        - you have images in data/ (or use Quick Demo)")
