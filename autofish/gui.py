from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from .capture import WindowCapture
from .config import AutoFishConfig, pick_default_model_path, resolve_model_path
from .input_controller import InputMode, SmartInputController
from .osc_input import OscInputSink
from .vision import YoloVision
from .win32_api import list_visible_windows
from .worker import AutoFishWorker


class AutoFishApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("VRChat Auto Fishing")
        self.geometry("840x560")

        self._worker: AutoFishWorker | None = None
        self._log_q: queue.Queue[str] = queue.Queue()
        self._status_var = tk.StringVar(value="idle")
        self._mode_var = tk.StringVar(value="message")
        self._window_var = tk.StringVar()
        self._model_var = tk.StringVar(value=str(pick_default_model_path(Path(__file__).resolve().parents[1])))
        self._osc_host_var = tk.StringVar(value="127.0.0.1")
        self._osc_port_var = tk.StringVar(value="9000")
        self._windows: list[tuple[int, str]] = []

        self._build_ui()
        self.after(120, self._drain_logs)
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
        top.columnconfigure(1, weight=1)

        btns = ttk.Frame(root)
        btns.pack(fill=tk.X, pady=(10, 6))
        ttk.Button(btns, text="Start", command=self.start_worker).pack(side=tk.LEFT)
        ttk.Button(btns, text="Stop", command=self.stop_worker).pack(side=tk.LEFT, padx=8)
        ttk.Label(btns, text="State:").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Label(btns, textvariable=self._status_var).pack(side=tk.LEFT)
        ttk.Label(btns, text="Input:").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Label(btns, textvariable=self._mode_var).pack(side=tk.LEFT)

        self.log_box = tk.Text(root, height=24, state=tk.DISABLED)
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def pick_model(self) -> None:
        got = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if got:
            self._model_var.set(got)

    def refresh_windows(self) -> None:
        self._windows = list_visible_windows()
        titles = [f"{title} (HWND={hwnd})" for hwnd, title in self._windows]
        self.window_combo["values"] = titles
        if titles and not self._window_var.get():
            self._window_var.set(titles[0])

    def _selected_hwnd(self) -> int:
        selected = self._window_var.get()
        for hwnd, title in self._windows:
            text = f"{title} (HWND={hwnd})"
            if text == selected:
                return hwnd
        return 0

    def start_worker(self) -> None:
        if self._worker is not None:
            self._log("worker already running")
            return
        hwnd = self._selected_hwnd()
        if not hwnd:
            self._log("no target window selected")
            return
        cfg = AutoFishConfig(
            osc_host=self._osc_host_var.get().strip() or "127.0.0.1",
            osc_port=int(self._osc_port_var.get().strip() or "9000"),
        )
        model = resolve_model_path(self._model_var.get(), Path(__file__).resolve().parents[1])
        detector = YoloVision(str(model), conf_yolo0=cfg.conf_yolo0, conf_yolo1=cfg.conf_yolo1)
        self._log(f"model loaded: {model}")
        self._log(f"model classes: {detector.model.names}")
        if 0 not in detector.model.names or 1 not in detector.model.names:
            self._log("warning: model missing class 0 or 1, detection flow may fail")
        capture = WindowCapture(hwnd=hwnd)
        sink = OscInputSink(cfg=cfg)
        input_ctl = SmartInputController(sink=sink, retry_limit=cfg.input_retry_limit, start_mode=InputMode.MESSAGE)
        self._worker = AutoFishWorker(
            cfg=cfg,
            detector=detector,
            capture=capture,
            input_ctl=input_ctl,
            log_cb=self._log_q.put,
            status_cb=self._status_var.set,
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
