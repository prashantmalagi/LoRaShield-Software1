"""
LoRaShield – Dashboard Page
"""

import tkinter as tk
from tkinter import ttk
import time
import threading

from utils import serial_comm
from utils.history import record_count
from utils.ai import is_model_loaded
from utils.encryption import is_crypto_available


# ---------------------------------------------------------------------------
# Color constants (shared across UI pages)
# ---------------------------------------------------------------------------
C = {
    'bg':          '#0D1117',
    'sidebar':     '#010409',
    'card':        '#161B22',
    'card2':       '#1C2128',
    'border':      '#30363D',
    'accent':      '#2563EB',
    'accent_h':    '#1D4ED8',
    'accent_glow': '#3B82F6',
    'text1':       '#E6EDF3',
    'text2':       '#8B949E',
    'text3':       '#656D76',
    'success':     '#3FB950',
    'warning':     '#D29922',
    'error':       '#F85149',
    'purple':      '#8B5CF6',
    'teal':        '#14B8A6',
}


def _make_card(parent: tk.Widget, row: int, col: int,
               padx: tuple = (8, 8), pady: tuple = (8, 8)) -> tk.Frame:
    card = tk.Frame(parent, bg=C['card'], bd=0,
                    highlightbackground=C['border'], highlightthickness=1)
    card.grid(row=row, column=col, padx=padx, pady=pady, sticky='nsew')
    return card


class StatusIndicator(tk.Frame):
    """A labeled status LED with animated pulse."""

    def __init__(self, parent, label: str, color: str = C['success'], **kw):
        super().__init__(parent, bg=C['card2'], **kw)
        self._color = color
        self._pulse_dir = 1
        self._alpha = 255

        # LED Canvas
        self.canvas = tk.Canvas(self, width=14, height=14,
                                bg=C['card2'], bd=0, highlightthickness=0)
        self.canvas.pack(side='left', padx=(10, 6), pady=8)
        self._led = self.canvas.create_oval(2, 2, 12, 12, fill=color, outline='')

        tk.Label(self, text=label, bg=C['card2'],
                 fg=C['text1'], font=('Segoe UI', 10)).pack(side='left', pady=8)

        self._pulse()

    def _pulse(self):
        # Simple brightness oscillation by blending color with bg
        self.canvas.itemconfig(self._led, fill=self._color)
        self.after(800, self._pulse)

    def set_status(self, color: str, label_text: str | None = None):
        self._color = color
        self.canvas.itemconfig(self._led, fill=color)
        if label_text is not None:
            for w in self.winfo_children():
                if isinstance(w, tk.Label):
                    w.config(text=label_text)


class MetricCard(tk.Frame):
    """A single metric/KPI card with value and description."""

    def __init__(self, parent, title: str, value: str,
                 sub: str, color: str = C['accent'], icon_char: str = '', **kw):
        super().__init__(parent, bg=C['card'],
                         highlightbackground=C['border'],
                         highlightthickness=1, **kw)
        # Icon / accent stripe at top
        stripe = tk.Frame(self, bg=color, height=3)
        stripe.pack(fill='x')

        body = tk.Frame(self, bg=C['card'], padx=18, pady=16)
        body.pack(fill='both', expand=True)

        # Icon char
        if icon_char:
            tk.Label(body, text=icon_char, bg=C['card'],
                     fg=color, font=('Segoe UI', 22)).pack(anchor='w')

        self.value_label = tk.Label(body, text=value, bg=C['card'],
                                    fg=C['text1'], font=('Segoe UI', 30, 'bold'))
        self.value_label.pack(anchor='w', pady=(4, 0))

        tk.Label(body, text=title, bg=C['card'],
                 fg=C['text2'], font=('Segoe UI', 11, 'bold')).pack(anchor='w')

        self.sub_label = tk.Label(body, text=sub, bg=C['card'],
                                   fg=C['text3'], font=('Segoe UI', 9))
        self.sub_label.pack(anchor='w', pady=(2, 0))

        # Hover effect
        self.bind('<Enter>', lambda e: self.config(highlightbackground=color))
        self.bind('<Leave>', lambda e: self.config(highlightbackground=C['border']))

    def update_value(self, new_val: str):
        self.value_label.config(text=new_val)

    def update_sub(self, new_sub: str):
        self.sub_label.config(text=new_sub)


class DashboardPage(tk.Frame):
    """Dashboard page showing system status and key metrics."""

    # Indices for status_leds list
    _IDX_CONN      = 0
    _IDX_AI        = 1
    _IDX_ESP32     = 2
    _IDX_LORA      = 3
    _IDX_ENC       = 4
    _IDX_PORT      = 5
    _IDX_SIGNAL    = 6
    _IDX_BATTERY   = 7

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._status_leds: list[tuple[tk.Canvas, int, tk.Label]] = []
        self._build()
        self._start_clock()
        self._start_status_poll()

    def _build(self):
        # ── Page header ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))

        tk.Label(hdr, text='SYSTEM DASHBOARD', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')

        self.clock_label = tk.Label(hdr, text='', bg=C['bg'],
                                    fg=C['text3'], font=('Consolas', 10))
        self.clock_label.pack(side='right')

        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=28, pady=(10, 0))

        # ── Status grid ──────────────────────────────────────────────────────
        sec1_lbl = tk.Label(self, text='SYSTEM STATUS', bg=C['bg'],
                            fg=C['text3'], font=('Segoe UI', 9, 'bold'))
        sec1_lbl.pack(anchor='w', padx=28, pady=(16, 6))

        grid_frame = tk.Frame(self, bg=C['bg'])
        grid_frame.pack(fill='x', padx=20)

        status_items = [
            ('Connection Status', C['warning'], 'Disconnected'),
            ('AI Model Status',   C['error'],   'Not Loaded'),
            ('ESP32 Status',      C['error'],   'Offline'),
            ('LoRa Status',       C['error'],   'Offline'),
            ('Encryption Status', C['success'], 'AES-256 Ready'),
            ('Current COM Port',  C['text2'],   'None'),
            ('Signal Strength',   C['warning'], 'N/A'),
            ('Battery Status',    C['success'],  'N/A'),
        ]

        self._status_leds = []
        cols = 4
        for i, (label, color, _status) in enumerate(status_items):
            row_idx = i // cols
            col_idx = i % cols
            grid_frame.rowconfigure(row_idx, weight=1)
            grid_frame.columnconfigure(col_idx, weight=1)

            cell = tk.Frame(grid_frame, bg=C['card2'],
                            highlightbackground=C['border'],
                            highlightthickness=1)
            cell.grid(row=row_idx, column=col_idx, padx=6, pady=6, sticky='ew')

            inner = tk.Frame(cell, bg=C['card2'])
            inner.pack(fill='x', padx=10, pady=8)

            canvas = tk.Canvas(inner, width=10, height=10,
                               bg=C['card2'], bd=0, highlightthickness=0)
            canvas.pack(side='left', padx=(0, 8))
            led_id = canvas.create_oval(1, 1, 9, 9, fill=color, outline='')

            lbl = tk.Label(inner, text=label, bg=C['card2'],
                           fg=C['text2'], font=('Segoe UI', 9))
            lbl.pack(side='left')

            # Value label (right-aligned)
            val_lbl = tk.Label(inner, text=_status, bg=C['card2'],
                               fg=color, font=('Segoe UI', 8, 'bold'))
            val_lbl.pack(side='right', padx=(0, 4))

            self._status_leds.append((canvas, led_id, val_lbl))

        # ── Metric cards ─────────────────────────────────────────────────────
        sec2_lbl = tk.Label(self, text='KEY METRICS', bg=C['bg'],
                            fg=C['text3'], font=('Segoe UI', 9, 'bold'))
        sec2_lbl.pack(anchor='w', padx=28, pady=(20, 6))

        cards_row = tk.Frame(self, bg=C['bg'])
        cards_row.pack(fill='x', padx=20, pady=(0, 16))
        for i in range(4):
            cards_row.columnconfigure(i, weight=1)

        metrics = [
            ('Messages Sent',      '0',     'Total outbound packets', C['accent'],   '[>]'),
            ('Messages Received',  '0',     'Total inbound packets',  C['success'],  '[<]'),
            ('AI Accuracy',        'N/A',   'Model not loaded',       C['purple'],   '[*]'),
            ('Device Status',      'READY', 'All systems nominal',    C['teal'],     '[#]'),
        ]

        self.metric_cards: list[MetricCard] = []
        for col, (title, val, sub, color, icon) in enumerate(metrics):
            card = MetricCard(cards_row, title=title, value=val,
                              sub=sub, color=color, icon_char=icon)
            card.grid(row=0, column=col, padx=6, pady=0, sticky='nsew')
            self.metric_cards.append(card)

        # ── Activity log ─────────────────────────────────────────────────────
        sec3_lbl = tk.Label(self, text='ACTIVITY LOG', bg=C['bg'],
                            fg=C['text3'], font=('Segoe UI', 9, 'bold'))
        sec3_lbl.pack(anchor='w', padx=28, pady=(8, 6))

        log_frame = tk.Frame(self, bg=C['card'],
                             highlightbackground=C['border'], highlightthickness=1)
        log_frame.pack(fill='both', expand=True, padx=26, pady=(0, 20))

        self.log_box = tk.Text(log_frame, bg=C['card'], fg=C['text2'],
                               font=('Consolas', 9), bd=0, padx=12, pady=10,
                               state='disabled', wrap='word',
                               insertbackground=C['accent'],
                               selectbackground=C['accent'],
                               highlightthickness=0)
        sb = ttk.Scrollbar(log_frame, orient='vertical',
                           command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        self._log('LoRaShield system initialized.')
        self._log(f'AES-256 encryption: {"Ready" if is_crypto_available() else "NOT available (install pycryptodome)"}.')
        self._log('Waiting for device connection...')

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.log_box.configure(state='normal')
        self.log_box.insert('end', f'[{ts}] {msg}\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _set_led(self, idx: int, color: str, value_text: str) -> None:
        """Update a status-grid LED and its value label."""
        if idx >= len(self._status_leds):
            return
        canvas, led_id, val_lbl = self._status_leds[idx]
        canvas.itemconfig(led_id, fill=color)
        val_lbl.config(text=value_text, fg=color)

    def _start_clock(self):
        def tick():
            now = time.strftime('%Y-%m-%d  %H:%M:%S')
            self.clock_label.config(text=now)
            self.after(1000, tick)
        tick()

    def _start_status_poll(self):
        """Poll backend state every 2 s and refresh dashboard indicators."""
        def poll():
            try:
                self._refresh_status()
            except Exception:
                pass
            self.after(2000, poll)
        self.after(2000, poll)

    def _refresh_status(self):
        """Read current backend state and update all status LEDs and metric cards."""
        connected = serial_comm.is_connected()
        ai_loaded  = is_model_loaded()
        enc_ready  = is_crypto_available()
        port       = self.app_state.get('com_port', '')
        stats      = serial_comm.get_stats()
        total      = record_count()
        sent       = stats.get('msgs_sent', 0)
        received   = stats.get('msgs_received', 0)
        last_comm  = stats.get('last_comm_time', 'Never')

        # Connection
        if connected:
            self._set_led(self._IDX_CONN,  C['success'], 'Connected')
            self._set_led(self._IDX_ESP32, C['success'], 'Online')
            self._set_led(self._IDX_LORA,  C['success'], 'Active')
        else:
            self._set_led(self._IDX_CONN,  C['warning'], 'Disconnected')
            self._set_led(self._IDX_ESP32, C['error'],   'Offline')
            self._set_led(self._IDX_LORA,  C['error'],   'Offline')

        # AI model
        if ai_loaded:
            self._set_led(self._IDX_AI, C['success'], 'AI Loaded')
        else:
            self._set_led(self._IDX_AI, C['error'],   'Not Loaded')

        # Encryption
        if enc_ready:
            self._set_led(self._IDX_ENC, C['success'], 'AES-256 Ready')
        else:
            self._set_led(self._IDX_ENC, C['error'],   'Not Available')

        # COM port
        port_display = port if port else 'None'
        self._set_led(self._IDX_PORT, C['text2'], port_display)

        # Signal and battery — hardware-dependent, shown as N/A when disconnected
        if not connected:
            self._set_led(self._IDX_SIGNAL,  C['warning'], 'N/A')
            self._set_led(self._IDX_BATTERY, C['text3'],   'N/A')

        # Metric cards
        self.metric_cards[0].update_value(str(sent))
        self.metric_cards[0].update_sub(f'Last comm: {last_comm}')
        self.metric_cards[1].update_value(str(received))
        self.metric_cards[1].update_sub(f'Total records: {total}')
        if ai_loaded:
            self.metric_cards[2].update_value('ON')
            self.metric_cards[2].update_sub('AI model ready')
        else:
            self.metric_cards[2].update_value('N/A')
            self.metric_cards[2].update_sub('Model not loaded')

        device_ok = connected and enc_ready
        self.metric_cards[3].update_value('READY' if device_ok else 'IDLE')
        self.metric_cards[3].update_sub('Connected & encrypted' if device_ok else 'Awaiting connection')

    # ── Public API (called from app.py navigation) ───────────────────────────

    def update_metric(self, idx: int, value: str):
        """Update a metric card value (0=Sent, 1=Received, 2=Accuracy, 3=Device)."""
        if 0 <= idx < len(self.metric_cards):
            self.metric_cards[idx].update_value(value)

    def add_log(self, message: str):
        """Append a line to the activity log."""
        self._log(message)
