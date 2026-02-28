from __future__ import annotations

import queue
import tkinter as tk
import time
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .capture import WindowCapture
from .config import AutoFishConfig, pick_default_model_path, resolve_model_path
from .input_controller import InputMode, SmartInputController
from .gui_logic import choose_vrchat_candidates, state_to_cn
from .osc_input import OscInputSink
from .vision import YoloVision
from .win32_api import Win32InputSink, list_visible_windows
from .worker import AutoFishWorker


class AutoFishApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VRChat Auto Fishing")
        self.geometry("840x560")

        self._worker: AutoFishWorker | None = None
        self._log_q: queue.Queue[str] = queue.Queue()
        self._preview_q: queue.Queue[tuple[object, object]] = queue.Queue(maxsize=1)
        self._fps_q: queue.Queue[tuple[float, float]] = queue.Queue(maxsize=2)
        self._status_var = tk.StringVar(value="idle")
        self._mode_var = tk.StringVar(value="message")
        self._fps_var = tk.StringVar(value="loop:0.0 infer:0.0 preview:0.0")
        self._window_var = tk.StringVar()
        self._model_var = tk.StringVar(value=str(pick_default_model_path(Path(__file__).resolve().parents[1])))
        self._osc_host_var = tk.StringVar(value="127.0.0.1")
        self._osc_port_var = tk.StringVar(value="9000")
        self._conf0_var = tk.StringVar(value="0.75")
        self._conf1_var = tk.StringVar(value="0.75")
        self._infer_fps_var = tk.StringVar(value="60")
        self._loop_fps_var = tk.StringVar(value="60")
        self._imgsz_var = tk.StringVar(value="640")
        self._mini_dead_px_var = tk.StringVar(value="3.0")
        self._mini_far_px_var = tk.StringVar(value="22.0")
        self._mini_predict_ms_var = tk.StringVar(value="100")
        self._mini_vel_alpha_var = tk.StringVar(value="0.35")
        self._mini_edge_guard_px_var = tk.StringVar(value="2.5")
        self._mini_brake_ms_var = tk.StringVar(value="90")
        self._mini_hold_track_ms_var = tk.StringVar(value="120")
        self._mini_hold_catch_ms_var = tk.StringVar(value="70")
        self._mini_track_px_ref_var = tk.StringVar(value="90")
        self._mini_up_full_ms_var = tk.StringVar(value="700")
        self._mini_hold_min_ms_var = tk.StringVar(value="150")
        self._mini_hold_max_ms_var = tk.StringVar(value="260")
        self._mini_drop_need_px_var = tk.StringVar(value="3.0")
        self._mini_wait_max_ms_var = tk.StringVar(value="1200")
        self._mini_signal_timeout_ms_var = tk.StringVar(value="100")
        self._roi_lock_delay_ms_var = tk.StringVar(value="500")
        self._windows: list[tuple[int, str]] = []
        self._yolo_imgtk = None
        self._roi_imgtk = None
        self._yolo_preview_win = None
        self._roi_preview_win = None
        self._yolo_preview_label = None
        self._roi_preview_label = None
        self._preview_count = 0
        self._preview_last_ts = 0.0
        self._preview_fps = 0.0
        self._loop_fps_actual = 0.0
        self._infer_fps_actual = 0.0

        self._build_ui()
        self._build_preview_windows()
        self.after(120, self._drain_logs)
        self.after(16, self._drain_previews)
        self.after(120, self._drain_fps)
        self.refresh_windows()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)

        top = ttk.Frame(root)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Target Window").grid(row=0, column=0, sticky="w")
        self.window_combo = ttk.Combobox(top, textvariable=self._window_var, width=60, state="readonly")
        self.window_combo.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Refresh", command=self.refresh_windows).grid(row=0, column=2, padx=4)

        ttk.Label(top, text="Model").grid(row=1, column=0, sticky="w")
        ttk.Entry(top, textvariable=self._model_var, width=62).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Browse", command=self.pick_model).grid(row=1, column=2, padx=4)

        ttk.Label(top, text="OSC").grid(row=2, column=0, sticky="w")
        osc_row = ttk.Frame(top)
        osc_row.grid(row=2, column=1, sticky="ew", padx=8)
        ttk.Entry(osc_row, textvariable=self._osc_host_var, width=24).pack(side=tk.LEFT)
        ttk.Label(osc_row, text=":").pack(side=tk.LEFT, padx=4)
        ttk.Entry(osc_row, textvariable=self._osc_port_var, width=8).pack(side=tk.LEFT)

        ttk.Label(top, text="YOLO").grid(row=3, column=0, sticky="w")
        yolo_row = ttk.Frame(top)
        yolo_row.grid(row=3, column=1, sticky="ew", padx=8)
        ttk.Label(yolo_row, text="conf0").pack(side=tk.LEFT)
        ttk.Entry(yolo_row, textvariable=self._conf0_var, width=6).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(yolo_row, text="conf1").pack(side=tk.LEFT)
        ttk.Entry(yolo_row, textvariable=self._conf1_var, width=6).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(yolo_row, text="inferFPS").pack(side=tk.LEFT)
        ttk.Entry(yolo_row, textvariable=self._infer_fps_var, width=6).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(yolo_row, text="loopFPS").pack(side=tk.LEFT)
        ttk.Entry(yolo_row, textvariable=self._loop_fps_var, width=6).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(yolo_row, text="imgsz").pack(side=tk.LEFT)
        ttk.Entry(yolo_row, textvariable=self._imgsz_var, width=7).pack(side=tk.LEFT, padx=(4, 0))

        mini = ttk.LabelFrame(top, text="MiniGame")
        mini.grid(row=4, column=1, sticky="ew", padx=8, pady=(8, 0))
        fields = [
            ("deadPx", self._mini_dead_px_var),
            ("farPx", self._mini_far_px_var),
            ("predictMs", self._mini_predict_ms_var),
            ("velAlpha", self._mini_vel_alpha_var),
            ("edgePx", self._mini_edge_guard_px_var),
            ("brakeMs", self._mini_brake_ms_var),
            ("intTrack", self._mini_hold_track_ms_var),
            ("intCatch", self._mini_hold_catch_ms_var),
            ("trackPx", self._mini_track_px_ref_var),
            ("upFullMs", self._mini_up_full_ms_var),
            ("holdMin", self._mini_hold_min_ms_var),
            ("holdMax", self._mini_hold_max_ms_var),
            ("dropNeed", self._mini_drop_need_px_var),
            ("waitMax", self._mini_wait_max_ms_var),
            ("sigTout", self._mini_signal_timeout_ms_var),
            ("roiLock", self._roi_lock_delay_ms_var),
        ]
        for i, (name, var) in enumerate(fields):
            r = i // 8
            c = (i % 8) * 2
            ttk.Label(mini, text=name).grid(row=r, column=c, sticky="w", padx=(2, 2), pady=2)
            ttk.Entry(mini, textvariable=var, width=6).grid(row=r, column=c + 1, sticky="w", padx=(0, 6), pady=2)
        top.columnconfigure(1, weight=1)

        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, pady=(10, 6))
        ttk.Button(btns, text="Start", command=self.start_worker).pack(side=tk.LEFT)
        ttk.Button(btns, text="Stop", command=self.stop_worker).pack(side=tk.LEFT, padx=8)
        ttk.Label(btns, text="State:").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Label(btns, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(btns, text="Input:").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Label(btns, textvariable=self._mode_var).pack(side=tk.LEFT)
        ttk.Label(btns, text="FPS:").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Label(btns, textvariable=self._fps_var).pack(side=tk.LEFT)

        self.log_box = tk.Text(root, height=16, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def _build_preview_windows(self) -> None:
        self._yolo_preview_win = tk.Toplevel(self)
        self._yolo_preview_win.title("YOLO 实时预览")
        self._yolo_preview_win.geometry("900x540")
        self._yolo_preview_label = ttk.Label(self._yolo_preview_win, text="YOLO预览初始化中")
        self._yolo_preview_label.pack(fill=tk.BOTH, expand=True)

        self._roi_preview_win = tk.Toplevel(self)
        self._roi_preview_win.title("OpenCV ROI 预览")
        self._roi_preview_win.geometry("460x540")
        self._roi_preview_label = ttk.Label(self._roi_preview_win, text="ROI预览初始化中")
        self._roi_preview_label.pack(fill=tk.BOTH, expand=True)

    def pick_model(self) -> None:
        got = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if got:
            self._model_var.set(got)

    def refresh_windows(self) -> None:
        self._windows = list_visible_windows()
        titles = [f"{title} (HWND={hwnd})" for hwnd, title in self._windows]
        self.window_combo["values"] = titles
        self._auto_pick_vrchat()

    def _auto_pick_vrchat(self) -> None:
        matches = choose_vrchat_candidates(self._windows)
        if not matches:
            self._log("未找到标题包含 VRChat 的窗口")
            return
        if len(matches) == 1:
            hwnd, title = matches[0]
            self._window_var.set(f"{title} (HWND={hwnd})")
            self._log(f"自动选中窗口: {title}")
            return
        picked = self._choose_window_dialog(matches)
        if picked is not None:
            hwnd, title = picked
            self._window_var.set(f"{title} (HWND={hwnd})")
            self._log(f"已选择窗口: {title}")

    def _choose_window_dialog(self, candidates: list[tuple[int, str]]) -> tuple[int, str] | None:
        dlg = tk.Toplevel(self)
        dlg.title("选择 VRChat 窗口")
        dlg.geometry("700x320")
        dlg.transient(self)
        dlg.grab_set()
        ttk.Label(dlg, text="检测到多个 VRChat 窗口，请选择一个：").pack(anchor="w", padx=12, pady=(10, 6))
        lb = tk.Listbox(dlg, height=12)
        lb.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))
        for hwnd, title in candidates:
            lb.insert(tk.END, f"{title} (HWND={hwnd})")
        lb.selection_set(0)
        result = {"idx": None}

        def _ok():
            sel = lb.curselection()
            if sel:
                result["idx"] = sel[0]
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        btn = ttk.Frame(dlg)
        btn.pack(fill=tk.X, padx=12, pady=(0, 10))
        ttk.Button(btn, text="确定", command=_ok).pack(side=tk.LEFT)
        ttk.Button(btn, text="取消", command=_cancel).pack(side=tk.LEFT, padx=8)
        dlg.wait_window()
        if result["idx"] is None:
            return None
        return candidates[result["idx"]]

    def _selected_target(self) -> tuple[int, str]:
        selected = self._window_var.get()
        for hwnd, title in self._windows:
            text = f"{title} (HWND={hwnd})"
            if text == selected:
                return hwnd, title
        return 0, ""

    def start_worker(self) -> None:
        if self._worker is not None:
            self._log("worker already running")
            return
        if not self._window_var.get():
            self._auto_pick_vrchat()
        hwnd, title = self._selected_target()
        if not hwnd:
            self._log("no target window selected")
            return
        try:
            cfg = AutoFishConfig(
                osc_host=self._osc_host_var.get().strip() or "127.0.0.1",
                osc_port=int(self._osc_port_var.get().strip() or "9000"),
                conf_yolo0=float(self._conf0_var.get().strip() or "0.5"),
                conf_yolo1=float(self._conf1_var.get().strip() or "0.5"),
                infer_fps=int(self._infer_fps_var.get().strip() or "10"),
                loop_fps=int(self._loop_fps_var.get().strip() or "20"),
                imgsz=int(self._imgsz_var.get().strip() or "640"),
                mini_dead_px=float(self._mini_dead_px_var.get().strip() or "3.0"),
                mini_far_px=float(self._mini_far_px_var.get().strip() or "22.0"),
                mini_predict_ms=int(self._mini_predict_ms_var.get().strip() or "140"),
                mini_vel_alpha=float(self._mini_vel_alpha_var.get().strip() or "0.35"),
                mini_edge_guard_px=float(self._mini_edge_guard_px_var.get().strip() or "2.5"),
                mini_brake_ms=int(self._mini_brake_ms_var.get().strip() or "140"),
                mini_hold_interval_track_ms=int(self._mini_hold_track_ms_var.get().strip() or "70"),
                mini_hold_interval_catch_ms=int(self._mini_hold_catch_ms_var.get().strip() or "45"),
                mini_track_px_ref=float(self._mini_track_px_ref_var.get().strip() or "90"),
                mini_up_full_ms=float(self._mini_up_full_ms_var.get().strip() or "700"),
                mini_hold_min_ms=int(self._mini_hold_min_ms_var.get().strip() or "150"),
                mini_hold_max_ms=int(self._mini_hold_max_ms_var.get().strip() or "350"),
                mini_drop_need_px=float(self._mini_drop_need_px_var.get().strip() or "3.0"),
                mini_wait_max_ms=int(self._mini_wait_max_ms_var.get().strip() or "1200"),
                mini_signal_timeout_ms=int(self._mini_signal_timeout_ms_var.get().strip() or "100"),
                roi_lock_delay_ms=int(self._roi_lock_delay_ms_var.get().strip() or "500"),
            )
        except ValueError:
            messagebox.showerror("参数错误", "YOLO/OSC 参数格式不正确，请检查数字输入。")
            return
        model = resolve_model_path(self._model_var.get(), Path(__file__).resolve().parents[1])
        detector = YoloVision(str(model), conf_yolo0=cfg.conf_yolo0, conf_yolo1=cfg.conf_yolo1)
        self._log(f"model loaded: {model}")
        self._log(f"model classes: {detector.model.names}")
        if 0 not in detector.model.names or 1 not in detector.model.names:
            self._log("warning: model missing class 0 or 1, detection flow may fail")
        capture = WindowCapture(hwnd=hwnd, window_name=title)
        win32_sink = Win32InputSink(hwnd=hwnd)
        sink = OscInputSink(cfg=cfg, win32_sink=win32_sink)
        input_ctl = SmartInputController(sink=sink, retry_limit=cfg.input_retry_limit, start_mode=InputMode.MESSAGE)
        self._worker = AutoFishWorker(
            cfg=cfg,
            detector=detector,
            capture=capture,
            input_ctl=input_ctl,
            log_cb=self._log_q.put,
            status_cb=lambda s: self._status_var.set(state_to_cn(s)),
            preview_cb=self._push_preview,
            fps_cb=self._push_fps,
        )
        self._worker.start()
        self._mode_var.set("osc")
        self._log("start requested")

    def stop_worker(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker = None
        self._status_var.set("idle")
        self._log("stop requested")

    def _log(self, text: str) -> None:
        self._log_q.put(text)

    def _drain_logs(self) -> None:
        while not self._log_q.empty():
            msg = self._log_q.get_nowait()
            self.log_box.configure(state=tk.NORMAL)
            self.log_box.insert(tk.END, msg + "\n")
            self.log_box.see(tk.END)
            self.log_box.configure(state=tk.DISABLED)
        self.after(120, self._drain_logs)

    def _drain_previews(self) -> None:
        latest = None
        while not self._preview_q.empty():
            latest = self._preview_q.get_nowait()
        if latest is not None:
            yolo_frame, roi_frame = latest
            if yolo_frame is not None:
                self._yolo_imgtk = self._to_imgtk(yolo_frame, 860, 500)
                if self._yolo_preview_label is not None and self._yolo_preview_label.winfo_exists():
                    self._yolo_preview_label.configure(image=self._yolo_imgtk, text="")
            if roi_frame is not None:
                self._roi_imgtk = self._to_imgtk(roi_frame, 430, 500)
                if self._roi_preview_label is not None and self._roi_preview_label.winfo_exists():
                    self._roi_preview_label.configure(image=self._roi_imgtk, text="")
            self._preview_count += 1
        t = time.time()
        if self._preview_last_ts <= 0.0:
            self._preview_last_ts = t
        elif t - self._preview_last_ts >= 1.0:
            self._preview_fps = self._preview_count / (t - self._preview_last_ts)
            self._preview_count = 0
            self._preview_last_ts = t
            self._refresh_fps_text()
        self.after(16, self._drain_previews)

    def _push_preview(self, yolo_frame, roi_frame) -> None:
        try:
            self._preview_q.put_nowait((yolo_frame, roi_frame))
            return
        except queue.Full:
            pass
        try:
            _ = self._preview_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._preview_q.put_nowait((yolo_frame, roi_frame))
        except queue.Full:
            pass

    def _push_fps(self, loop_fps: float, infer_fps: float) -> None:
        try:
            self._fps_q.put_nowait((loop_fps, infer_fps))
            return
        except queue.Full:
            pass
        try:
            _ = self._fps_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._fps_q.put_nowait((loop_fps, infer_fps))
        except queue.Full:
            pass

    def _drain_fps(self) -> None:
        latest = None
        while not self._fps_q.empty():
            latest = self._fps_q.get_nowait()
        if latest is not None:
            self._loop_fps_actual, self._infer_fps_actual = latest
            self._refresh_fps_text()
        self.after(120, self._drain_fps)

    def _refresh_fps_text(self) -> None:
        self._fps_var.set(
            f"loop:{self._loop_fps_actual:.1f} "
            f"infer:{self._infer_fps_actual:.1f} "
            f"preview:{self._preview_fps:.1f}"
        )

    @staticmethod
    def _to_imgtk(bgr_frame, max_w: int, max_h: int):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(max_w / max(1, w), max_h / max(1, h), 1.0)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=image)
