"""
LoRaShield – Settings Page
"""

import tkinter as tk
from tkinter import ttk
import json
import os
from pathlib import Path

C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950', 'error': '#F85149',
}

_BASE_DIR     = Path(__file__).resolve().parent.parent
SETTINGS_FILE = _BASE_DIR / 'data' / 'settings.json'


def _btn(parent, text, command, color=None, fg=None, **kw):
    color = color or C['accent']
    fg = fg or C['text1']
    b = tk.Button(parent, text=text, command=command, bg=color, fg=fg,
                  font=('Segoe UI', 10, 'bold'), bd=0, padx=16, pady=8,
                  cursor='hand2', activebackground=C['accent_h'],
                  activeforeground=C['text1'], relief='flat', **kw)
    b.bind('<Enter>', lambda e: b.config(bg=C['accent_h'] if color == C['accent'] else color))
    b.bind('<Leave>', lambda e: b.config(bg=color))
    return b


def _row(parent, label: str, widget_factory, default=None):
    """Build a settings row: label on left, widget on right."""
    row = tk.Frame(parent, bg=C['card2'],
                   highlightbackground=C['border'], highlightthickness=1)
    row.pack(fill='x', pady=4)

    lbl_frame = tk.Frame(row, bg=C['card2'], width=200)
    lbl_frame.pack(side='left', fill='y')
    lbl_frame.pack_propagate(False)
    tk.Label(lbl_frame, text=label, bg=C['card2'],
             fg=C['text2'], font=('Segoe UI', 10), anchor='w'
             ).pack(side='left', padx=16, pady=12)

    ctrl_frame = tk.Frame(row, bg=C['card2'])
    ctrl_frame.pack(side='left', fill='x', expand=True, padx=16, pady=8)

    widget = widget_factory(ctrl_frame)
    widget.pack(side='left')
    return widget


class SettingsPage(tk.Frame):
    """Application Settings page."""

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._vars: dict[str, tk.Variable] = {}
        self._build()
        self._load_settings()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        tk.Label(hdr, text='SETTINGS', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=28, pady=(10, 0))

        # Scrollable body
        canvas = tk.Canvas(self, bg=C['bg'], bd=0, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        body = tk.Frame(canvas, bg=C['bg'])
        canvas.create_window((0, 0), window=body, anchor='nw')
        body.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        def mw(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', mw)

        def section(title: str) -> tk.Frame:
            tk.Label(body, text=title, bg=C['bg'],
                     fg=C['text3'], font=('Segoe UI', 9, 'bold')
                     ).pack(anchor='w', padx=28, pady=(20, 6))
            f = tk.Frame(body, bg=C['bg'])
            f.pack(fill='x', padx=28)
            return f

        # ── Interface ─────────────────────────────────────────────────────────
        iface = section('INTERFACE')

        self._vars['theme'] = tk.StringVar(value='Dark')
        _row(iface, 'Theme',
             lambda p: ttk.Combobox(p, textvariable=self._vars['theme'],
                                    values=['Dark', 'Dark Blue', 'Dark Slate'],
                                    width=18, state='readonly'))

        # ── Hardware ──────────────────────────────────────────────────────────
        hw = section('HARDWARE')

        self._vars['com_port'] = tk.StringVar(value=self.app_state.get('com_port', ''))
        _row(hw, 'Default COM Port',
             lambda p: tk.Entry(p, textvariable=self._vars['com_port'],
                                bg=C['card'], fg=C['text1'], font=('Segoe UI', 10),
                                bd=0, insertbackground=C['accent'], width=14,
                                highlightthickness=1,
                                highlightbackground=C['border']))

        self._vars['baud_rate'] = tk.StringVar(value='9600')
        _row(hw, 'Baud Rate',
             lambda p: ttk.Combobox(p, textvariable=self._vars['baud_rate'],
                                    values=['9600', '19200', '38400', '57600', '115200'],
                                    width=12, state='readonly'))

        self._vars['auto_connect'] = tk.BooleanVar(value=False)

        def make_toggle(p):
            cb = tk.Checkbutton(p, variable=self._vars['auto_connect'],
                                bg=C['card2'], fg=C['text1'],
                                selectcolor=C['accent'], activebackground=C['card2'],
                                font=('Segoe UI', 10), bd=0, cursor='hand2')
            return cb

        _row(hw, 'Auto Connect on Start', make_toggle)

        # ── AI Settings ───────────────────────────────────────────────────────
        ai = section('AI CONFIGURATION')

        self._vars['ai_threshold'] = tk.DoubleVar(value=0.75)

        def make_scale(p):
            f = tk.Frame(p, bg=C['card2'])
            scale = tk.Scale(f, variable=self._vars['ai_threshold'],
                             from_=0.0, to=1.0, resolution=0.01,
                             orient='horizontal', length=200,
                             bg=C['card2'], fg=C['text1'],
                             troughcolor=C['border'],
                             activebackground=C['accent'],
                             highlightthickness=0, bd=0)
            scale.pack(side='left')
            lbl = tk.Label(f, bg=C['card2'], fg=C['accent'],
                           font=('Consolas', 10))
            lbl.pack(side='left', padx=8)

            def update_lbl(*_):
                lbl.config(text=f'{self._vars["ai_threshold"].get():.2f}')
            self._vars['ai_threshold'].trace_add('write', update_lbl)
            update_lbl()
            return f

        _row(ai, 'Confidence Threshold', make_scale)

        # ── Security ──────────────────────────────────────────────────────────
        sec = section('SECURITY')

        self._vars['enc_key'] = tk.StringVar(value=self.app_state.get('enc_key', ''))

        def make_key_entry(p):
            e = tk.Entry(p, textvariable=self._vars['enc_key'],
                         show='*', bg=C['card'], fg=C['text1'],
                         font=('Consolas', 10), bd=0, width=32,
                         insertbackground=C['accent'],
                         highlightthickness=1, highlightbackground=C['border'])
            return e

        _row(sec, 'Default Encryption Key', make_key_entry)

        # ── Save button ───────────────────────────────────────────────────────
        btn_row = tk.Frame(body, bg=C['bg'])
        btn_row.pack(fill='x', padx=28, pady=(24, 8))
        _btn(btn_row, '  SAVE SETTINGS  ', self._save, C['accent']).pack(side='left', padx=(0, 12))
        _btn(btn_row, '  RESET DEFAULTS  ', self._reset, '#21262D', C['text1']).pack(side='left')

        self.status_var = tk.StringVar(value='')
        tk.Label(btn_row, textvariable=self.status_var, bg=C['bg'],
                 fg=C['success'], font=('Segoe UI', 9)).pack(side='left', padx=16)

        tk.Frame(body, bg=C['bg'], height=30).pack()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self):
        data = {k: v.get() for k, v in self._vars.items()}
        # Propagate to app_state
        self.app_state.update({
            'com_port':    data.get('com_port', ''),
            'enc_key':     data.get('enc_key', ''),
            'baud_rate':   int(data.get('baud_rate', 9600)),
            'auto_connect': bool(data.get('auto_connect', False)),
        })
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with SETTINGS_FILE.open('w') as fh:
            json.dump(data, fh, indent=2)
        self.status_var.set('Settings saved successfully.')

    def _load_settings(self):
        if not SETTINGS_FILE.exists():
            return
        try:
            with SETTINGS_FILE.open() as fh:
                data = json.load(fh)
            for k, v in data.items():
                if k in self._vars:
                    self._vars[k].set(v)
        except Exception:
            pass

    def _reset(self):
        defaults = {
            'theme': 'Dark', 'com_port': '', 'baud_rate': '9600',
            'auto_connect': False, 'ai_threshold': 0.75, 'enc_key': '',
        }
        for k, v in defaults.items():
            if k in self._vars:
                self._vars[k].set(v)
        self.status_var.set('Defaults restored.')
