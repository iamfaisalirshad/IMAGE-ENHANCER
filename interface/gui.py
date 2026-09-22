"""
IMPORTANT: do not import this module at startup of main.py.
It is only imported when the user asks for the GUI, so that the
terminal menu works even if Tkinter is unavailable.
"""

import os
import queue
import threading

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import config
from utils.device import get_device, device_summary
from utils.file_utils import ensure_dirs, save_result_image


def _load_model(name, device):
    from inference.predictor import load_model
    return load_model(name, device)


def _anczos():
    # Pillow >= 10 exposes Image.Resampling; keep a fallback for older ones.
    return getattr(Image, "Resampling", Image).LANCZOS


class EnhancerApp:
    def __init__(self, root):
        self.root = root
        self.device = get_device()
        self.width = config.IMAGE_SIZE
        self.out = config.TARGET_SIZE
        self.path = None
        self.model_name = "v2"
        self.model = None
        self.preview = None
        self.in_img = None
        self.out_img = None
        self._result_queue = queue.Queue()

        self._build()
        self._refresh_model_status()
        self.root.bind("<Configure>", self._on_resize)

    def _build(self):
        self.root.title("IMAGE ENHANCER")
        self.root.geometry("980x720")
        self.root.minsize(640, 480)

        tk.Label(self.root, text="IMAGE ENHANCER",
                 font=("Arial", 18, "bold")).pack(pady=6)
        tk.Label(self.root, text="AI-Based Image Restoration and "
                                 "2x Super-Resolution",
                 font=("Arial", 11)).pack()
        tk.Label(self.root, text=f"Device: {device_summary()}",
                 font=("Arial", 9), fg="gray").pack()

        # Model selection
        sel = tk.Frame(self.root)
        sel.pack(pady=8)
        tk.Label(sel, text="Select Model:").pack(side="left")
        self.model_var = tk.StringVar(value="V2")
        v1_btn = tk.Radiobutton(sel, text="V1", variable=self.model_var,
                                value="V1", command=self._model_changed)
        v2_btn = tk.Radiobutton(sel, text="V2", variable=self.model_var,
                                value="V2", command=self._model_changed)
        v1_btn.pack(side="left", padx=6)
        v2_btn.pack(side="left", padx=6)

        # Buttons
        btns = tk.Frame(self.root)
        btns.pack(pady=6)
        tk.Button(btns, text="Upload Image",
                  command=self.upload).pack(side="left", padx=6)
        tk.Button(btns, text="Enhance",
                  command=self.enhance).pack(side="left", padx=6)
        tk.Button(btns, text="Save Result", command=self.save).pack(
            side="left", padx=6)

        # Status label
        self.status = tk.Label(self.root, text="", fg="#1a6f1a")
        self.status.pack()

        # Metrics
        self.metric_label = tk.Label(self.root, text="", font=("Arial", 10))
        self.metric_label.pack(pady=4)

        # Two halves filling the window: input (left) | enhanced (right)
        self.view = tk.Frame(self.root)
        self.view.pack(fill="both", expand=True, padx=6, pady=4)

        self.in_pane = tk.Frame(self.view, bg="#222222", relief="sunken", bd=1)
        self.out_pane = tk.Frame(self.view, bg="#222222", relief="sunken", bd=1)
        self.in_pane.pack(side="left", fill="both", expand=True, padx=(0, 3))
        self.out_pane.pack(side="left", fill="both", expand=True, padx=(3, 0))

        self.in_label = tk.Label(self.in_pane, text="Input (uploaded)",
                                 fg="#ffffff", bg="#222222")
        self.out_label = tk.Label(self.out_pane, text="Enhanced (2x)",
                                  fg="#ffffff", bg="#222222")
        self.in_label.pack(fill="both", expand=True)
        self.out_label.pack(fill="both", expand=True)

        captions = tk.Frame(self.root)
        captions.pack(fill="x", pady=(0, 4))
        tk.Label(captions, text="Uploaded image",
                 font=("Arial", 10, "bold")).pack(side="left", expand=True)
        tk.Label(captions, text="Enhanced image",
                 font=("Arial", 10, "bold")).pack(side="left", expand=True)

    # -- layout -------------------------------------------------------------
    def _on_resize(self, event):
        if event.widget is self.root:
            self._render_images()

    def _render_images(self):
        self.root.update_idletasks()
        try:
            pane_w = max(self.in_pane.winfo_width() - 2, 40)
            pane_h = max(self.in_pane.winfo_height() - 2, 40)
        except tk.TclError:
            return
        for label, img in ((self.in_label, self.in_img),
                           (self.out_label, self.out_img)):
            if label.winfo_exists() == 0:
                continue
            if img is None:
                label.config(image="", text="No image loaded", font=("Arial", 12))
            else:
                disp = self._fit(img, pane_w, pane_h)
                photo = ImageTk.PhotoImage(disp)
                label.config(image=photo, text="", width=0, height=0)
                label.image = photo

    @staticmethod
    def _fit(img, max_w, max_h):
        img = img.convert("RGB")
        w, h = img.size
        scale = min(max_w / w, max_h / h)
        scale = min(scale, 1.0 if max(img.size) < 800 else 2.0)
        if scale < 1.0 or scale > 2.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             _anczos())
        return img

    # -- model management ------------------------------------------------
    def _model_changed(self):
        self.model_name = self.model_var.get().lower()
        self.model = None
        self._refresh_model_status()

    def _refresh_model_status(self):
        name = self.model_name
        ckpt = config.CHECKPOINT_V1 if name == "v1" else config.CHECKPOINT_V2
        if os.path.isfile(ckpt):
            self.status.config(text=f"{name.upper()} model ready.",
                               fg="#1a6f1a")
        else:
            self.status.config(text=f"No trained {name.upper()} model. "
                                    f"Train it from the terminal menu.",
                               fg="#a5521a")

    # -- actions -----------------------------------------------------------
    def upload(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        self.path = path
        self.preview = None
        self.out_img = None
        self.metric_label.config(text="")
        img = Image.open(path).convert("RGB")
        self.in_img = img
        self._render_images()
        self.status.config(text="Image loaded. Click Enhance.",
                           fg="#1a6f1a")

    def _ensure_model(self):
        if self.model is not None:
            return True
        name = self.model_name
        ckpt = config.CHECKPOINT_V1 if name.lower() == "v1" else config.CHECKPOINT_V2
        if not os.path.isfile(ckpt):
            messagebox.showerror(
                "Model Missing",
                f"No trained {name.upper()} model found.\n\n"
                "Please train it first using the terminal menu "
                "(option 2 in main.py).",
            )
            return False
        self.model = _load_model(name.lower(), self.device)
        return True

    def enhance(self):
        if not self.path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        if not self._ensure_model():
            return

        self.status.config(text="Enhancing (2x super-resolution)... "
                                "please wait",
                           fg="#1a6f1a")
        self.root.update_idletasks()

        def work():
            try:
                pil = Image.open(self.path).convert("RGB")
                input_2x, enhanced = self.model.enhance_2x(pil)
                self._result_queue.put(("ok", pil.size, input_2x, enhanced))
            except Exception as e:
                self._result_queue.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()
        # Poll the queue from the main loop (thread-safe Tkinter update).
        self._poll_results()

    def _poll_results(self):
        try:
            item = self._result_queue.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll_results)
            return
        if item[0] == "error":
            messagebox.showerror("Enhance Error",
                                 f"Could not enhance: {item[1]}")
            self.status.config(text="Enhance failed.", fg="#a00")
            return
        _, orig_size, input_2x, enhanced = item
        self.preview = Image.fromarray(enhanced)
        self.out_img = self.preview
        self._render_images()
        w, h = orig_size
        self.status.config(
            text=f"Enhanced with {self.model_name.upper()}: "
                 f"{w}x{h} -> {2*w}x{2*h}. Click 'Save Result' to keep it.",
            fg="#1a6f1a")

    def save(self):
        if self.preview is None:
            messagebox.showwarning("Nothing to Save",
                                   "Enhance an image first.")
            return
        saved = save_result_image(np.array(self.preview), config.ENHANCED_DIR,
                                  prefix="enhanced_gui")
        self.metric_label.config(text=f"Saved: {saved}")
        self.status.config(text="Result saved.", fg="#1a6f1a")


def launch():
    root = tk.Tk()
    ensure_dirs(config.REQUIRED_DIRS)
    EnhancerApp(root)
    root.mainloop()
