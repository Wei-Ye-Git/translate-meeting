import tkinter as tk
from tkinter import ttk

from i18n import t


class Overlay:
    def __init__(self, alpha=0.92, font_size=18, max_history_records=5000):
        self.root = tk.Tk()
        self.root.title(t("app.title"))
        self.root.attributes("-alpha", alpha)
        self.root.configure(bg="#101216")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        screen_width = self.root.winfo_screenwidth()
        self.window_width = max(560, min(1300, int(screen_width * 0.78)))
        self.padding = 14
        self.wrap = self.window_width - self.padding * 2
        self.history = []
        self.history_metadata = []
        self.history_offset = 0
        self.max_history_records = max_history_records
        self.english_records = []
        self.all_english = []
        self.current_english = ""
        self.current_translation = ""
        self.live_buffer = ""
        self.history_widgets = []
        self.closed = False
        self.history_at_bottom = True
        self.font_size = font_size
        self.topmost = False

        self._drag_offset = (0, 0)
        self._build_ui(font_size)
        self.root.geometry(f"{self.window_width}x560+{(screen_width - self.window_width) // 2}+80")

    def _build_ui(self, font_size):
        header = tk.Frame(self.root, bg="#1b1e24")
        header.pack(fill="x")
        self.status = tk.Label(
            header, text=t("ui.waiting_audio"), fg="#99a2ad", bg="#1b1e24",
            font=("Microsoft YaHei UI", 10), anchor="w",
        )
        self.status.pack(side="left", padx=self.padding, pady=7)
        self.copy_bilingual_btn = tk.Button(
            header, text=t("ui.copy_bilingual"), command=self.copy_bilingual,
            fg="#d7dde5", bg="#303640", activeforeground="#ffffff",
            activebackground="#48515e", relief="flat", borderwidth=0,
            font=("Microsoft YaHei UI", 9), padx=8, pady=3,
        )
        self.copy_bilingual_btn.pack(side="right", padx=(4, self.padding), pady=4)
        self.copy_btn = tk.Button(
            header, text=t("ui.copy_english"), command=self.copy_english,
            fg="#d7dde5", bg="#303640", activeforeground="#ffffff",
            activebackground="#48515e", relief="flat", borderwidth=0,
            font=("Microsoft YaHei UI", 9), padx=8, pady=3,
        )
        self.copy_btn.pack(side="right", padx=4, pady=4)
        self.top_toggle = tk.Button(
            header, text=t("ui.pin"), command=self.toggle_topmost,
            fg="#d7dde5", bg="#2c3340", activeforeground="#ffffff",
            activebackground="#414d60", relief="flat", borderwidth=0,
            font=("Microsoft YaHei UI", 9), padx=8, pady=3,
        )
        self.top_toggle.pack(side="right", padx=4, pady=4)
        tk.Label(
            header, text=t("ui.drag_window"), fg="#68717d", bg="#1b1e24",
            font=("Microsoft YaHei UI", 9),
        ).pack(side="right", padx=4)
        for widget in (header, self.status):
            widget.bind("<ButtonPress-1>", self._start_move)
            widget.bind("<B1-Motion>", self._on_move)

        history_frame = tk.Frame(self.root, bg="#20242b", highlightthickness=1,
                                 highlightbackground="#353b45")
        history_frame.pack(fill="both", expand=True, padx=self.padding, pady=(8, 10))
        history_frame.pack_propagate(False)
        history_frame.configure(height=230)
        tk.Label(
            history_frame, text=t("ui.history_title"), fg="#9ba5b2", bg="#20242b",
            font=("Microsoft YaHei UI", 10), anchor="w",
        ).pack(fill="x", padx=10, pady=(7, 3))

        scrollbar = ttk.Scrollbar(history_frame, orient="vertical")
        self.history_text = tk.Text(
            history_frame,
            bg="#20242b",
            fg="#f1f3f5",
            wrap="word",
            highlightthickness=0,
            borderwidth=0,
            padx=12,
            pady=6,
            font=("Microsoft YaHei UI", 11),
            cursor="arrow",
            yscrollcommand=self._on_scroll,
        )
        scrollbar.config(command=self.history_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.history_text.pack(side="left", fill="both", expand=True)

        self.history_text.tag_configure("en", foreground="#abb4bf", font=("Segoe UI", 10))
        self.history_text.tag_configure("zh", foreground="#f1f3f5", font=("Microsoft YaHei UI", 12))
        self.history_scrollbar = scrollbar
        self.history_text.bind("<Key>", self._on_text_key)

        current = tk.Frame(self.root, bg="#101216", highlightthickness=1,
                           highlightbackground="#303640")
        current.pack(fill="x", padx=self.padding, pady=(0, self.padding))
        self.current_label = tk.Label(
            current, text=t("ui.live_preview"), fg="#8994a1", bg="#101216",
            font=("Microsoft YaHei UI", 9), anchor="w",
        )
        self.current_label.pack(fill="x", padx=10, pady=(7, 0))
        self.original = tk.Label(
            current, text="", fg="#ffffff", bg="#101216",
            font=("Segoe UI", max(12, font_size - 2)), wraplength=self.wrap - 20,
            justify="left", anchor="w",
        )
        self.original.pack(fill="x", padx=10, pady=(2, 0))
        self.translation = tk.Label(
            current, text="", fg="#9ba5b2", bg="#101216",
            font=("Microsoft YaHei UI", max(10, font_size - 4)), wraplength=self.wrap - 20,
            justify="left", anchor="w",
        )
        self.translation.pack(fill="x", padx=10, pady=(1, 7))
        self.root.bind("<Double-Button-1>", lambda event: self.close() if event.widget in (self.root, self.status) else None)

    def _on_text_key(self, event):
        if event.keysym in ("Up", "Down", "Prior", "Next", "Home", "End", "Left", "Right"):
            return None
        if (event.state & 4) and event.keysym.lower() in ("c", "a"):
            return None
        return "break"

    def _on_scroll(self, first, last):
        self.history_scrollbar.set(first, last)
        self.history_at_bottom = float(last) >= 0.98

    def _start_move(self, event):
        self._drag_offset = (event.x, event.y)

    def _on_move(self, event):
        x = self.root.winfo_x() + event.x - self._drag_offset[0]
        y = self.root.winfo_y() + event.y - self._drag_offset[1]
        self.root.geometry(f"+{x}+{y}")

    def set_status(self, text):
        self.status.config(text=text)

    def update_current(self, original, translated=""):
        self.current_english = original or ""
        self.current_translation = translated or ""
        self.original.config(text=self.current_english)
        self.translation.config(text=self.current_translation)

    def append_english(self, text):
        if text and text.strip():
            cleaned = text.strip()
            if self.live_buffer == cleaned:
                return
            self.live_buffer = cleaned
            if not self.all_english or self.all_english[-1] != cleaned:
                self.all_english.append(cleaned)
            self.update_current(cleaned, "")

    def replace_history(self, items):
        metadata = list(self.history_metadata)
        old_offset = self.history_offset
        self.history = []
        self.history_metadata = []
        self.history_offset = old_offset
        self.history_text.delete("1.0", "end")
        for idx, (orig, trans) in enumerate(items):
            entry_metadata = metadata[idx] if idx < len(metadata) else None
            self._append_history_entry(idx, orig, trans, metadata=entry_metadata)

    def append_history(self, index, original, translated, metadata=None):
        self._append_history_entry(index, original, translated, metadata)

    def _append_history_entry(self, index, original, translated, metadata=None):
        self.history.append((original, translated))
        index = len(self.history) - 1
        metadata = metadata or {}
        self.history_metadata.append(metadata)
        if len(self.history) > self.max_history_records:
            self.history.pop(0)
            self.history_metadata.pop(0)
            self.history_offset += 1
            self.replace_history(list(self.history))
            return
        en_tag = f"en_{index}"
        zh_tag = f"zh_{index}"
        zh_display = translated if translated else t("ui.translating")
        start = metadata.get("start")
        end = metadata.get("end")
        prefix = (
            f"[{self._format_timestamp(start)}-{self._format_timestamp(end)}] "
            if start is not None and end is not None else ""
        )
        corrected = t("ui.corrected_marker") if metadata.get("corrected") else ""
        self.history_text.insert(
            "end", f"{prefix}{corrected}{original}\n", ("en", en_tag)
        )
        self.history_text.insert("end", f"{zh_display}\n\n", ("zh", zh_tag))
        if self.history_at_bottom:
            self.history_text.see("end")

    def update_history(self, index, translated, version=None):
        if index >= len(self.history):
            return False
        metadata = self.history_metadata[index]
        if version is not None and version < metadata.get("version", 1):
            return False
        if index < len(self.history):
            original, _ = self.history[index]
            self.history[index] = (original, translated)
        zh_tag = f"zh_{index}"
        ranges = self.history_text.tag_ranges(zh_tag)
        if ranges:
            self.history_text.delete(ranges[0], ranges[1])
            self.history_text.insert(ranges[0], f"{translated}\n\n", ("zh", zh_tag))
        return True

    def update_history_original(self, index, original, metadata=None, version=None):
        if index >= len(self.history):
            return False
        if metadata:
            current_version = self.history_metadata[index].get("version", 1)
            new_version = metadata.get("version", current_version)
        else:
            current_version = self.history_metadata[index].get("version", 1)
            new_version = version if version is not None else current_version
        if version is not None and version < current_version:
            return False
        if new_version < current_version:
            return False
        if index < len(self.history):
            old_translation = self.history[index][1]
            self.history[index] = (original, "")
            if metadata:
                previous_original = self.history_metadata[index].get("previous_original")
                previous_translation = self.history_metadata[index].get("previous_translation")
                self.history_metadata[index] = dict(metadata)
                if old_translation and old_translation != t("ui.translating"):
                    self.history_metadata[index].setdefault(
                        "previous_translation", old_translation
                    )
                if previous_original:
                    self.history_metadata[index].setdefault("previous_original", previous_original)
                if previous_translation:
                    self.history_metadata[index].setdefault("previous_translation", previous_translation)
        en_tag = f"en_{index}"
        ranges = self.history_text.tag_ranges(en_tag)
        if ranges:
            current_metadata = self.history_metadata[index]
            start = current_metadata.get("start")
            end = current_metadata.get("end")
            prefix = (
                f"[{self._format_timestamp(start)}-{self._format_timestamp(end)}] "
                if start is not None and end is not None else ""
            )
            self.history_text.delete(ranges[0], ranges[1])
            corrected = t("ui.corrected_marker") if current_metadata.get("corrected") else ""
            self.history_text.insert(
                ranges[0], f"{prefix}{corrected}{original}\n", ("en", en_tag)
            )
        self.history_metadata[index]["version"] = new_version
        self.update_history(index, t("ui.translating"), version=new_version)
        return True

    def mark_history_for_review(self, index):
        if index >= len(self.history):
            return False
        self.history_metadata[index]["review"] = True
        en_tag = f"en_{index}"
        ranges = self.history_text.tag_ranges(en_tag)
        if ranges:
            metadata = self.history_metadata[index]
            start = metadata.get("start")
            end = metadata.get("end")
            prefix = (
                f"[{self._format_timestamp(start)}-{self._format_timestamp(end)}] "
                if start is not None and end is not None else ""
            )
            original = self.history[index][0]
            corrected = t("ui.corrected_marker") if metadata.get("corrected") else ""
            self.history_text.delete(ranges[0], ranges[1])
            self.history_text.insert(
                ranges[0], f"{prefix}{t('ui.review_marker')}{corrected}{original}\n", ("en", en_tag)
            )
        return True

    @staticmethod
    def _format_timestamp(seconds):
        minutes, seconds = divmod(max(0.0, seconds), 60.0)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        return f"{minutes:02d}:{seconds:06.3f}"

    def toggle_topmost(self):
        self.topmost = not self.topmost
        self.root.attributes("-topmost", self.topmost)
        self.top_toggle.config(text=t("ui.unpin") if self.topmost else t("ui.pin"))
        self.set_status(t("ui.window_pinned") if self.topmost else t("ui.window_unpinned"))

    def copy_bilingual(self):
        text = self.bilingual_text()
        if not text:
            self.set_status(t("ui.no_bilingual_history"))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self.set_status(t("ui.copied_bilingual", count=len(self.history)))

    def bilingual_text(self):
        lines = []
        for index, (original, translated) in enumerate(self.history):
            if not original.strip():
                continue
            metadata = self.history_metadata[index] if index < len(self.history_metadata) else {}
            start = metadata.get("start")
            end = metadata.get("end")
            prefix = (
                f"[{self._format_timestamp(start)}-{self._format_timestamp(end)}] "
                if start is not None and end is not None else ""
            )
            review = t("ui.review_marker") if metadata.get("review") else ""
            corrected = t("ui.corrected_marker") if metadata.get("corrected") else ""
            lines.append(f"{prefix}{review}{corrected}{t('ui.english_label')}{original.strip()}")
            previous_original = metadata.get("previous_original")
            previous_translation = metadata.get("previous_translation")
            if previous_original:
                lines.append(t("export.revision_english") + previous_original.strip())
            if previous_translation:
                lines.append(t("export.revision_chinese") + previous_translation.strip())
            if translated and translated.strip():
                lines.append(t("ui.chinese_label") + translated.strip())
            lines.append("")
        return "\n".join(lines).strip()

    def copy_english(self):
        lines = []
        for index, (original, _) in enumerate(self.history):
            if not original.strip():
                continue
            metadata = self.history_metadata[index] if index < len(self.history_metadata) else {}
            start = metadata.get("start")
            end = metadata.get("end")
            prefix = (
                f"[{self._format_timestamp(start)}-{self._format_timestamp(end)}] "
                if start is not None and end is not None else ""
            )
            review = t("ui.review_marker") if metadata.get("review") else ""
            corrected = t("ui.corrected_marker") if metadata.get("corrected") else ""
            lines.append(f"{prefix}{review}{corrected}{original.strip()}")
        records = lines
        text = "\n".join(records)
        if not text:
            self.set_status(t("ui.no_english_history"))
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self.set_status(t("ui.copied_english", count=len(records)))

    def close(self):
        self.closed = True
        self.root.destroy()

    def run(self):
        self.root.mainloop()
