import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import os

from main import (
    convert_to_gray, binary_threshold, rotate_image, crop_image, zoom_image,
    rgb_to_yiq, rgb_to_cmyk, histogram_strecting, add_images, multiply_images,
    adjust_brightness, adaptive_threshold, sobel_edge_detection, add_to_noise,
    clear_noise_filter_with_mean, clear_noise_filter_with_median, gaussian_blur,
    create_structuring_element, dilate, erode, dilate_erode_open, erode_dilate_close,
)

BG        = "#1e1e2e"
SURFACE   = "#2a2a3e"
SURFACE2  = "#313244"
ACCENT    = "#7c3aed"
ACCENT_H  = "#6d28d9"
TEXT      = "#cdd6f4"
TEXT_DIM  = "#a6adc8"
SUCCESS   = "#a6e3a1"
BORDER    = "#45475a"
CANVAS_BG = "#111122"

CATEGORIES = {
    "Donusumler":      ["Gri Donusum", "Binary", "Parlaklik"],
    "Renk Uzayi":      ["CMYK", "YIQ"],
    "Gurultu":         ["Salt&Pepper Ekle", "Mean Filter", "Median Filter"],
    "Bulaniklik":      ["Gaussian Blur"],
    "Esikleme":        ["Adaptif Esikleme"],
    "Kenar Tespiti":   ["Sobel"],
    "Morfoloji":       ["Dilate", "Erode", "Opening", "Closing"],
    "Geometri":        ["Dondur", "Kirp", "Zoom"],
    "Histogram":       ["Histogram Germe"],
    "Aritmetik":       ["Goruntu Toplama", "Goruntu Carpma"],
}

LABEL_MAP = {
    "Donusumler":    "Dönüşümler",
    "Renk Uzayi":    "Renk Uzayı",
    "Gurultu":       "Gürültü",
    "Bulaniklik":    "Bulanıklaştırma",
    "Esikleme":      "Eşikleme",
    "Kenar Tespiti": "Kenar Tespiti",
    "Morfoloji":     "Morfoloji",
    "Geometri":      "Geometri",
    "Histogram":     "Histogram",
    "Aritmetik":     "Aritmetik",
    "Gri Donusum":        "Gri Dönüşüm",
    "Binary":             "Binary",
    "Parlaklik":          "Parlaklık",
    "CMYK":               "CMYK",
    "YIQ":                "YIQ",
    "Salt&Pepper Ekle":   "Salt & Pepper Ekle",
    "Mean Filter":        "Mean Filter",
    "Median Filter":      "Median Filter",
    "Gaussian Blur":      "Gaussian Blur",
    "Adaptif Esikleme":   "Adaptif Eşikleme",
    "Sobel":              "Sobel Kenar Tespiti",
    "Dilate":             "Dilate",
    "Erode":              "Erode",
    "Opening":            "Opening (Açma)",
    "Closing":            "Closing (Kapama)",
    "Dondur":             "Döndür",
    "Kirp":               "Kırp",
    "Zoom":               "Zoom",
    "Histogram Germe":    "Histogram Germe",
    "Goruntu Toplama":    "Görüntü Toplama",
    "Goruntu Carpma":     "Görüntü Çarpma",
}


def resize_for_processing(image):
    """
    Görüntüyü işlem öncesi maksimum 800px'e küçültür.
    Aspect ratio korunur.
    PIL kullanılır (altyapı işlemi).
    Returns: resized numpy array, scale_factor used
    """
    from PIL import Image as PILImage
    MAX_SIZE = 800

    if len(image.shape) == 3:
        h, w = image.shape[:2]
    else:
        h, w = image.shape

    if max(h, w) <= MAX_SIZE:
        return image, 1.0  # zaten küçük, dokunma

    scale = MAX_SIZE / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    if len(image.shape) == 3:
        pil_img = PILImage.fromarray(image)
    else:
        pil_img = PILImage.fromarray(image.astype(np.uint8))

    pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
    return np.array(pil_img), scale


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması")
        self.root.minsize(1200, 700)
        self.root.configure(bg=BG)

        self.original_image = None
        self.current_image  = None
        self.second_image   = None
        self.second_image_name = ""

        self._orig_photo = None
        self._curr_photo = None
        self._processing = False
        self._resize_job = None

        self.params = {}
        self.second_img_var = None

        self._setup_style()
        self._build_ui()
        self.root.bind("<Configure>", self._on_resize)
        self.root.after(400, self._update_displays)

    # ── Style ──────────────────────────────────────────────────────────────────
    def _setup_style(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure(".",
            background=BG, foreground=TEXT,
            fieldbackground=SURFACE2, font=("Segoe UI", 10),
            troughcolor=SURFACE2, selectbackground=ACCENT, selectforeground=TEXT)
        s.configure("TFrame", background=BG)
        s.configure("TLabel", background=BG, foreground=TEXT)
        s.configure("TSeparator", background=BORDER)
        s.configure("TCombobox",
            fieldbackground=SURFACE2, background=SURFACE2,
            foreground=TEXT, arrowcolor=ACCENT,
            bordercolor=BORDER, lightcolor=SURFACE2, darkcolor=SURFACE2)
        s.map("TCombobox",
            fieldbackground=[("readonly", SURFACE2)],
            selectbackground=[("readonly", ACCENT)],
            selectforeground=[("readonly", TEXT)])
        s.configure("TEntry",
            fieldbackground=SURFACE2, foreground=TEXT,
            bordercolor=BORDER, insertcolor=TEXT)
        s.configure("TButton",
            background=SURFACE2, foreground=TEXT,
            borderwidth=0, relief="flat", padding=(8, 6))
        s.map("TButton",
            background=[("active", BORDER), ("pressed", BORDER)])
        s.configure("Accent.TButton",
            background=ACCENT, foreground="white",
            borderwidth=0, relief="flat", padding=(10, 8),
            font=("Segoe UI", 11, "bold"))
        s.map("Accent.TButton",
            background=[("active", ACCENT_H), ("pressed", ACCENT_H),
                        ("disabled", SURFACE2)])
        s.configure("TScrollbar", background=SURFACE2, troughcolor=SURFACE,
            bordercolor=SURFACE, arrowcolor=TEXT_DIM)

    # ── Layout ─────────────────────────────────────────────────────────────────
    def _build_ui(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True)

        self.left_panel = tk.Frame(main, bg=SURFACE, width=300)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel.pack_propagate(False)

        sep = tk.Frame(main, bg=BORDER, width=2)
        sep.pack(side=tk.LEFT, fill=tk.Y)

        self.right_panel = tk.Frame(main, bg=BG)
        self.right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        lp = self.left_panel

        # Header bar
        hdr = tk.Frame(lp, bg=ACCENT, height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Görüntü İşleme", bg=ACCENT, fg="white",
                 font=("Segoe UI", 13, "bold")).pack(expand=True)

        # Scrollable inner area
        outer = tk.Frame(lp, bg=SURFACE)
        outer.pack(fill=tk.BOTH, expand=True)

        vscroll = ttk.Scrollbar(outer, orient="vertical")
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self._inner_canvas = tk.Canvas(outer, bg=SURFACE, highlightthickness=0,
                                       yscrollcommand=vscroll.set)
        self._inner_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vscroll.configure(command=self._inner_canvas.yview)

        self.sf = tk.Frame(self._inner_canvas, bg=SURFACE)
        win_id = self._inner_canvas.create_window((0, 0), window=self.sf, anchor="nw")

        self.sf.bind("<Configure>",
            lambda e: self._inner_canvas.configure(
                scrollregion=self._inner_canvas.bbox("all")))
        self._inner_canvas.bind("<Configure>",
            lambda e: self._inner_canvas.itemconfig(win_id, width=e.width))
        self._inner_canvas.bind_all("<MouseWheel>",
            lambda e: self._inner_canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._populate_left(self.sf)

    def _populate_left(self, sf):
        P = dict(padx=14, pady=3)

        # ── File buttons ──
        bf = tk.Frame(sf, bg=SURFACE)
        bf.pack(fill=tk.X, **P)

        self.load_btn = ttk.Button(bf, text="Goruntu Yukle",
                                   command=self._load_image)
        self.load_btn.pack(fill=tk.X, pady=(10, 4))

        self.reset_btn = ttk.Button(bf, text="Orijinale Don",
                                    command=self._reset_to_original,
                                    state=tk.DISABLED)
        self.reset_btn.pack(fill=tk.X, pady=4)

        self.save_btn = ttk.Button(bf, text="Sonucu Kaydet",
                                   command=self._save_result,
                                   state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, pady=(4, 10))

        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, padx=14, pady=4)

        # ── Category & Operation combos ──
        sel = tk.Frame(sf, bg=SURFACE)
        sel.pack(fill=tk.X, **P)

        self._lbl(sel, "KATEGORİ", dim=True, bold=True, small=True).pack(
            anchor=tk.W, pady=(8, 2))
        self.category_var = tk.StringVar()
        self.cat_combo = ttk.Combobox(sel, textvariable=self.category_var,
                                      values=list(CATEGORIES.keys()),
                                      state="readonly")
        self.cat_combo.pack(fill=tk.X, pady=(0, 6))
        self.cat_combo.bind("<<ComboboxSelected>>", self._on_category_change)

        self._lbl(sel, "İŞLEM", dim=True, bold=True, small=True).pack(
            anchor=tk.W, pady=(0, 2))
        self.operation_var = tk.StringVar()
        self.op_combo = ttk.Combobox(sel, textvariable=self.operation_var,
                                     state="readonly")
        self.op_combo.pack(fill=tk.X, pady=(0, 8))
        self.op_combo.bind("<<ComboboxSelected>>", self._on_operation_change)

        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, padx=14, pady=4)

        # ── Dynamic params ──
        self.param_frame = tk.Frame(sf, bg=SURFACE)
        self.param_frame.pack(fill=tk.X, **P)

        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, padx=14, pady=4)

        # ── Apply ──
        af = tk.Frame(sf, bg=SURFACE)
        af.pack(fill=tk.X, padx=14, pady=8)
        self.apply_btn = ttk.Button(af, text="Uygula", style="Accent.TButton",
                                    command=self._apply_operation,
                                    state=tk.DISABLED)
        self.apply_btn.pack(fill=tk.X)

        # ── Timing ──
        self.timing_var = tk.StringVar(value="")
        tk.Label(sf, textvariable=self.timing_var, bg=SURFACE, fg=SUCCESS,
                 font=("Segoe UI", 9, "italic")).pack(padx=14, anchor=tk.W)

        ttk.Separator(sf, orient="horizontal").pack(fill=tk.X, padx=14, pady=6)

        # ── Status ──
        self.status_var = tk.StringVar(value="Goruntu yuklemek icin dusmeye basin.")
        tk.Label(sf, textvariable=self.status_var, bg=SURFACE, fg=TEXT_DIM,
                 font=("Segoe UI", 9), wraplength=260,
                 justify=tk.LEFT).pack(padx=14, pady=(0, 14), anchor=tk.W)

        self.second_img_var = tk.StringVar(value="Ikinci goruntu secilmedi")

    def _lbl(self, parent, text, dim=False, bold=False, small=False):
        fg   = TEXT_DIM if dim else TEXT
        size = 8 if small else 10
        wt   = "bold" if bold else "normal"
        return tk.Label(parent, text=text, bg=SURFACE, fg=fg,
                        font=("Segoe UI", size, wt))

    def _build_right_panel(self):
        rp = self.right_panel
        panels = tk.Frame(rp, bg=BG)
        panels.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.orig_canvas, self.orig_info = self._img_panel(panels, "Orijinal Goruntu")
        self.curr_canvas, self.curr_info = self._img_panel(panels, "Guncel Goruntu")

    def _img_panel(self, parent, title):
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        tk.Label(frame, text=title, bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(pady=(10, 4))

        canvas = tk.Canvas(frame, bg=CANVAS_BG, highlightthickness=1,
                           highlightbackground=BORDER)
        canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        info_var = tk.StringVar(value="—")
        tk.Label(frame, textvariable=info_var, bg=SURFACE, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(pady=(0, 10))
        return canvas, info_var

    # ── Category / Op selection ────────────────────────────────────────────────
    def _on_category_change(self, _=None):
        ops = CATEGORIES.get(self.category_var.get(), [])
        self.op_combo["values"] = ops
        self.op_combo.set("")
        self._clear_params()
        self.apply_btn.config(state=tk.DISABLED)

    def _on_operation_change(self, _=None):
        self._build_params()
        if self.original_image is not None:
            self.apply_btn.config(state=tk.NORMAL)

    # ── Param widget builders ──────────────────────────────────────────────────
    def _clear_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.params.clear()

    def _scale_row(self, parent, label, key, from_, to, default, res=1.0):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill=tk.X, pady=3)
        var  = tk.DoubleVar(value=default)
        disp = tk.StringVar(value=self._fmt(default, res))
        self.params[key] = var

        hdr = tk.Frame(row, bg=SURFACE)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text=label, bg=SURFACE, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Label(hdr, textvariable=disp, bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 9, "bold"), width=7).pack(side=tk.RIGHT)

        tk.Scale(row, variable=var, from_=from_, to=to, orient=tk.HORIZONTAL,
                 resolution=res, bg=SURFACE, fg=TEXT, troughcolor=SURFACE2,
                 highlightthickness=0, sliderrelief="flat",
                 activebackground=ACCENT, showvalue=False,
                 command=lambda v: disp.set(self._fmt(float(v), res))
                 ).pack(fill=tk.X)
        return var

    def _combo_row(self, parent, label, key, values, default=None):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=label, bg=SURFACE, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor=tk.W)
        var = tk.StringVar(value=str(default if default is not None else values[0]))
        self.params[key] = var
        ttk.Combobox(row, textvariable=var,
                     values=[str(v) for v in values],
                     state="readonly").pack(fill=tk.X)
        return var

    def _entry_row(self, parent, label, key, default="0"):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill=tk.X, pady=3)
        tk.Label(row, text=label, bg=SURFACE, fg=TEXT_DIM,
                 font=("Segoe UI", 9)).pack(anchor=tk.W)
        var = tk.StringVar(value=default)
        self.params[key] = var
        ttk.Entry(row, textvariable=var).pack(fill=tk.X)
        return var

    @staticmethod
    def _fmt(v, res):
        return f"{v:.2f}" if res < 1 else str(int(v))

    def _build_params(self):
        self._clear_params()
        op = self.operation_var.get()
        pf = self.param_frame

        if op == "Binary":
            self._scale_row(pf, "Esik degeri (threshold)", "threshold", 0, 255, 128)

        elif op == "Parlaklik":
            self._scale_row(pf, "Parlaklik (beta)", "beta", -150, 150, 0)

        elif op == "Salt&Pepper Ekle":
            self._scale_row(pf, "Tuz orani (salt)", "salt_ratio", 0.01, 0.5, 0.05, 0.01)
            self._scale_row(pf, "Biber orani (pepper)", "pepper_ratio", 0.01, 0.5, 0.05, 0.01)

        elif op in ("Mean Filter", "Median Filter"):
            self._combo_row(pf, "Kernel boyutu", "kernel_size", [3, 5, 7, 9, 11], 3)

        elif op == "Gaussian Blur":
            self._combo_row(pf, "Kernel boyutu", "kernel_size", [3, 5, 7, 9, 11], 3)
            self._scale_row(pf, "Sigma", "sigma", 0.1, 5.0, 1.0, 0.1)

        elif op == "Adaptif Esikleme":
            self._combo_row(pf, "Kernel boyutu", "kernel_size", [3, 5, 7, 9, 11], 11)
            self._scale_row(pf, "C degeri", "C", 1, 20, 2)

        elif op in ("Dilate", "Erode", "Opening", "Closing"):
            self._combo_row(pf, "Yapi elemani boyutu", "se_size", [3, 5, 7, 9], 3)

        elif op == "Dondur":
            self._scale_row(pf, "Derece", "degree", 0, 360, 90)

        elif op == "Kirp":
            self._entry_row(pf, "X (sol piksel)", "x", "0")
            self._entry_row(pf, "Y (ust piksel)", "y", "0")
            self._entry_row(pf, "Genislik (w)", "w", "100")
            self._entry_row(pf, "Yukseklik (h)", "h", "100")

        elif op == "Zoom":
            self._scale_row(pf, "Olcek faktoru", "scale_factor", 0.1, 3.0, 1.5, 0.1)

        elif op == "Histogram Germe":
            self._scale_row(pf, "Min deger", "min_value", 0, 255, 0)
            self._scale_row(pf, "Max deger", "max_value", 0, 255, 255)

        elif op in ("Goruntu Toplama", "Goruntu Carpma"):
            row = tk.Frame(pf, bg=SURFACE)
            row.pack(fill=tk.X, pady=6)
            cur = self.second_image_name if self.second_image is not None else "Secilmedi"
            self.second_img_var.set(f"Ikinci goruntu: {cur}")
            tk.Label(row, textvariable=self.second_img_var, bg=SURFACE, fg=TEXT_DIM,
                     font=("Segoe UI", 8), wraplength=250).pack(anchor=tk.W, pady=(0, 4))
            ttk.Button(row, text="Ikinci Goruntu Sec",
                       command=self._load_second_image).pack(fill=tk.X)

        else:
            tk.Label(pf, text="Bu islem icin parametre gerekmez.",
                     bg=SURFACE, fg=TEXT_DIM,
                     font=("Segoe UI", 9, "italic")).pack(pady=8, anchor=tk.W)

    # ── File I/O ───────────────────────────────────────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            title="Goruntu Sec",
            filetypes=[("Goruntu", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("Tumuu", "*.*")])
        if not path:
            return
        try:
            arr = np.array(Image.open(path).convert("RGB"))
            resized, scale = resize_for_processing(arr)
            self.original_image = resized
            self.current_image  = resized.copy()
            self.reset_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            if self.operation_var.get():
                self.apply_btn.config(state=tk.NORMAL)
            self.timing_var.set("")
            if scale < 1.0:
                orig_h, orig_w = arr.shape[:2]
                new_h, new_w   = resized.shape[:2]
                self._set_status(
                    f"Yuklendi: {os.path.basename(path)}\n"
                    f"Görüntü {orig_w}×{orig_h} → {new_w}×{new_h}"
                    f" olarak ölçeklendirildi (işlem için)"
                )
            else:
                self._set_status(f"Yuklendi: {os.path.basename(path)}")
            self._update_displays()
        except Exception as e:
            messagebox.showerror("Yukleme Hatasi", str(e))

    def _load_second_image(self):
        path = filedialog.askopenfilename(
            title="Ikinci Goruntu Sec",
            filetypes=[("Goruntu", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                       ("Tuumu", "*.*")])
        if not path:
            return
        try:
            self.second_image      = np.array(Image.open(path).convert("RGB"))
            self.second_image_name = os.path.basename(path)
            self.second_img_var.set(f"Ikinci goruntu: {self.second_image_name}")
        except Exception as e:
            messagebox.showerror("Yukleme Hatasi", str(e))

    def _reset_to_original(self):
        if self.original_image is None:
            return
        self.current_image = self.original_image.copy()
        self.timing_var.set("")
        self._set_status("Orijinal goruntue donuldu.")
        self._update_displays()

    def _save_result(self):
        if self.current_image is None:
            return
        path = filedialog.asksaveasfilename(
            title="Sonucu Kaydet",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")])
        if not path:
            return
        try:
            arr = self.current_image
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr, "L" if arr.ndim == 2 else "RGB").save(path)
            self._set_status(f"Kaydedildi: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Kaydetme Hatasi", str(e))

    # ── Apply pipeline ─────────────────────────────────────────────────────────
    def _apply_operation(self):
        if self.original_image is None:
            messagebox.showwarning("Uyari", "Once bir goruntu yukleyin!")
            return
        op = self.operation_var.get()
        if not op:
            messagebox.showwarning("Uyari", "Bir islem secin!")
            return
        if self._processing:
            return

        resized, scale = resize_for_processing(self.current_image)
        if scale < 1.0:
            orig_h, orig_w = self.current_image.shape[:2]
            new_h, new_w   = resized.shape[:2]
            self.current_image = resized
            self._set_status(
                f"Görüntü {orig_w}×{orig_h} → {new_w}×{new_h}"
                f" olarak ölçeklendirildi (işlem için)"
            )

        self._processing = True
        self.apply_btn.config(state=tk.DISABLED, text="Isleniyor...")
        if scale >= 1.0:
            self._set_status(f"{LABEL_MAP.get(op, op)} uygulanıyor...")
        threading.Thread(target=self._run_op, args=(op,), daemon=True).start()

    def _run_op(self, op):
        t0 = time.time()
        try:
            result, channels = self._execute(op)
            ms = (time.time() - t0) * 1000
            self.root.after(0, self._finish_op, op, result, channels, ms)
        except ValueError as e:
            self.root.after(0, self._fail_op, str(e))
        except Exception as e:
            self.root.after(0, self._fail_op, f"Beklenmeyen hata:\n{e}")

    def _ensure_gray(self, arr):
        if arr.ndim == 3:
            gray = convert_to_gray(arr)
            self.root.after(0, self._set_status,
                            "Uyari: Goruntu otomatik olarak griye donusturuldu.")
            return np.clip(gray, 0, 255).astype(np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _execute(self, op):
        cur = self.current_image
        p   = self.params
        gi  = lambda k: int(float(p[k].get()))
        gf  = lambda k: float(p[k].get())
        channels = None

        if op == "Gri Donusum":
            if cur.ndim == 2:
                return cur, channels
            result = np.clip(convert_to_gray(cur), 0, 255).astype(np.uint8)

        elif op == "Binary":
            result = binary_threshold(self._ensure_gray(cur), gi("threshold"))

        elif op == "Parlaklik":
            result = adjust_brightness(cur, gi("beta"))

        elif op == "CMYK":
            if cur.ndim == 2:
                raise ValueError("CMYK donusumu icin RGB goruntu gereklidir!")
            C, M, Y, K = rgb_to_cmyk(cur)
            channels = {"type": "cmyk", "C": C, "M": M, "Y": Y, "K": K}
            result = cur

        elif op == "YIQ":
            if cur.ndim == 2:
                raise ValueError("YIQ donusumu icin RGB goruntu gereklidir!")
            Yc, Ic, Qc = rgb_to_yiq(cur)
            channels = {"type": "yiq", "Y": Yc, "I": Ic, "Q": Qc}
            result = cur

        elif op == "Salt&Pepper Ekle":
            result = add_to_noise(self._ensure_gray(cur),
                                  gf("salt_ratio"), gf("pepper_ratio"))

        elif op == "Mean Filter":
            result = clear_noise_filter_with_mean(self._ensure_gray(cur), gi("kernel_size"))

        elif op == "Median Filter":
            result = clear_noise_filter_with_median(self._ensure_gray(cur), gi("kernel_size"))

        elif op == "Gaussian Blur":
            result = gaussian_blur(self._ensure_gray(cur),
                                   gi("kernel_size"), gf("sigma"))

        elif op == "Adaptif Esikleme":
            result = adaptive_threshold(self._ensure_gray(cur),
                                        gi("kernel_size"), gi("C"))

        elif op == "Sobel":
            result = sobel_edge_detection(self._ensure_gray(cur))

        elif op == "Dilate":
            result = dilate(self._ensure_gray(cur),
                            create_structuring_element(gi("se_size")))

        elif op == "Erode":
            result = erode(self._ensure_gray(cur),
                           create_structuring_element(gi("se_size")))

        elif op == "Opening":
            result = dilate_erode_open(self._ensure_gray(cur),
                                       create_structuring_element(gi("se_size")))

        elif op == "Closing":
            result = erode_dilate_close(self._ensure_gray(cur),
                                        create_structuring_element(gi("se_size")))

        elif op == "Dondur":
            result = rotate_image(self._ensure_gray(cur), gf("degree"))

        elif op == "Kirp":
            try:
                x, y, w, h = gi("x"), gi("y"), gi("w"), gi("h")
            except (ValueError, KeyError):
                raise ValueError("Gecersiz kirpma degerleri! Tamsayi girin.")
            ih, iw = cur.shape[:2]
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError("Kirpma koordinatlari negatif veya sifir olamaz!")
            if x + w > iw or y + h > ih:
                raise ValueError(
                    f"Kirpma alani goruntu sinirlarini asiyor ({iw}x{ih})!")
            result = crop_image(cur, x, y, w, h)

        elif op == "Zoom":
            sf = gf("scale_factor")
            if sf <= 0:
                raise ValueError("Olcek faktoru pozitif olmalidir!")
            result = zoom_image(self._ensure_gray(cur), sf)

        elif op == "Histogram Germe":
            mn, mx = gi("min_value"), gi("max_value")
            if mn >= mx:
                raise ValueError("Min deger Max degerden kucuk olmalidir!")
            result = histogram_strecting(self._ensure_gray(cur), mn, mx)

        elif op in ("Goruntu Toplama", "Goruntu Carpma"):
            if self.second_image is None:
                raise ValueError("Lutfen ikinci goruntuyu secin!")
            g1 = self._ensure_gray(cur)
            g2 = self._ensure_gray(self.second_image)
            if g1.shape != g2.shape:
                g2 = np.array(
                    Image.fromarray(g2).resize(
                        (g1.shape[1], g1.shape[0]), Image.LANCZOS))
            result = (add_images(g1, g2)
                      if op == "Goruntu Toplama"
                      else multiply_images(g1, g2))

        else:
            raise ValueError(f"Bilinmeyen islem: {op}")

        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result, channels

    def _finish_op(self, op, result, channels, ms):
        self._processing = False
        self.apply_btn.config(state=tk.NORMAL, text="Uygula")
        self.timing_var.set(f"Islem: {ms:.1f} ms")

        if op not in ("CMYK", "YIQ"):
            self.current_image = result
            self._set_status(f"{LABEL_MAP.get(op, op)} basariyla uygulandir.")
        else:
            self._set_status(f"{op} kanallari gosteriliyor.")

        if channels:
            if channels["type"] == "cmyk":
                self._show_cmyk(channels)
            elif channels["type"] == "yiq":
                self._show_yiq(channels)

        self._update_displays()

    def _fail_op(self, msg):
        self._processing = False
        self.apply_btn.config(state=tk.NORMAL, text="Uygula")
        messagebox.showerror("Islem Hatasi", msg)
        self._set_status("Hata olustu.")

    # ── Display ────────────────────────────────────────────────────────────────
    def _update_displays(self):
        self._put_image(self.orig_canvas, self.original_image, is_orig=True)
        self._put_image(self.curr_canvas, self.current_image,  is_orig=False)

        def info(arr, placeholder):
            if arr is None:
                return placeholder
            mode = "RGB" if arr.ndim == 3 else "Gri"
            return f"{arr.shape[1]}x{arr.shape[0]}  |  {mode}  |  {arr.dtype}"

        self.orig_info.set(info(self.original_image, "Goruntu yuklenmedi"))
        self.curr_info.set(info(self.current_image,  "Henuz islem yapilmadi"))

    def _put_image(self, canvas, arr, is_orig):
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10:
            cw, ch = 400, 300

        canvas.delete("all")
        if arr is None:
            canvas.create_text(cw // 2, ch // 2,
                               text="Goruntu yok", fill=TEXT_DIM,
                               font=("Segoe UI", 11))
            return

        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr, "L" if arr.ndim == 2 else "RGB")
        iw, ih = img.size
        ratio = min(cw / iw, ch / ih)
        nw = max(1, int(iw * ratio))
        nh = max(1, int(ih * ratio))
        photo = ImageTk.PhotoImage(img.resize((nw, nh), Image.LANCZOS))
        canvas.create_image((cw - nw) // 2, (ch - nh) // 2,
                            anchor=tk.NW, image=photo)
        if is_orig:
            self._orig_photo = photo
        else:
            self._curr_photo = photo

    def _on_resize(self, _=None):
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(120, self._update_displays)

    def _set_status(self, msg):
        self.status_var.set(msg)

    # ── Channel windows ────────────────────────────────────────────────────────
    def _channel_window(self, title, channels, cols):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.configure(bg=BG)
        rows = (len(channels) + cols - 1) // cols
        win.geometry(f"{cols * 370}x{rows * 320 + 70}")
        win.minsize(cols * 280, rows * 240 + 70)

        tk.Label(win, text=title, bg=BG, fg=TEXT,
                 font=("Segoe UI", 13, "bold")).pack(pady=8)

        grid = tk.Frame(win, bg=BG)
        grid.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        for idx, (name, data) in enumerate(channels):
            r, c = divmod(idx, cols)
            cell = tk.Frame(grid, bg=SURFACE)
            cell.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
            grid.grid_columnconfigure(c, weight=1)
            grid.grid_rowconfigure(r, weight=1)

            tk.Label(cell, text=name, bg=SURFACE, fg=TEXT,
                     font=("Segoe UI", 10, "bold")).pack(pady=(8, 4))

            cv = tk.Canvas(cell, bg=CANVAS_BG, highlightthickness=0)
            cv.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 8))

            arr8 = np.clip(data * 255, 0, 255).astype(np.uint8)
            pil  = Image.fromarray(arr8, "L")

            def _show(canvas=cv, image=pil):
                canvas.update_idletasks()
                cw = canvas.winfo_width()  or 280
                ch = canvas.winfo_height() or 200
                iw, ih = image.size
                ratio = min(cw / iw, ch / ih)
                nw = max(1, int(iw * ratio))
                nh = max(1, int(ih * ratio))
                resized = image.resize((nw, nh), Image.LANCZOS)
                photo = ImageTk.PhotoImage(resized)
                canvas.delete("all")
                canvas.create_image((cw - nw) // 2, (ch - nh) // 2,
                                    anchor=tk.NW, image=photo)
                canvas._photo = photo

            cv.after(150, _show)

    def _show_cmyk(self, ch):
        self._channel_window("CMYK Kanallari", [
            ("C  —  Cyan",    ch["C"]),
            ("M  —  Magenta", ch["M"]),
            ("Y  —  Yellow",  ch["Y"]),
            ("K  —  Black",   ch["K"]),
        ], cols=2)

    def _show_yiq(self, ch):
        def norm(a):
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn) if mx != mn else np.zeros_like(a)

        self._channel_window("YIQ Kanallari", [
            ("Y  —  Parlaklik (Luma)",    ch["Y"]),
            ("I  —  Renk (In-phase)",     norm(ch["I"])),
            ("Q  —  Renk (Quadrature)",   norm(ch["Q"])),
        ], cols=3)


def main():
    root = tk.Tk()
    ImageProcessingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
