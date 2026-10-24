"""
LoRaShield – Text to Morse Encoder Page
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from utils.ai import encode_text
from utils.history import save_history


C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950', 'error': '#F85149', 'input_bg': '#0D1117',
}


def _btn(parent, text, command, color=None, fg=None, **kw):
    color = color or C['accent']
    fg = fg or C['text1']
    b = tk.Button(parent, text=text, command=command, bg=color, fg=fg,
                  font=('Segoe UI', 10, 'bold'), bd=0, padx=18, pady=8,
                  cursor='hand2', activebackground=C['accent_h'],
                  activeforeground=C['text1'], relief='flat', **kw)
    b.bind('<Enter>', lambda e: b.config(bg=C['accent_h'] if color == C['accent'] else color))
    b.bind('<Leave>', lambda e: b.config(bg=color))
    return b


class EncoderPage(tk.Frame):
    """Text to Morse Code encoder page."""

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        tk.Label(hdr, text='TEXT  →  MORSE CODE', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=28, pady=(10, 0))

        tk.Label(self, text='Convert plain text to standard Morse code.',
                 bg=C['bg'], fg=C['text2'], font=('Segoe UI', 10)
                 ).pack(anchor='w', padx=28, pady=(10, 0))

        # Main body
        body = tk.Frame(self, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=20, pady=16)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        # ── Left: Input ──────────────────────────────────────────────────────
        lbl_in = tk.Label(body, text='INPUT TEXT', bg=C['bg'],
                          fg=C['text3'], font=('Segoe UI', 9, 'bold'))
        lbl_in.grid(row=0, column=0, sticky='w', padx=8, pady=(0, 4))

        in_frame = tk.Frame(body, bg=C['card'],
                            highlightbackground=C['border'], highlightthickness=1)
        in_frame.grid(row=1, column=0, sticky='nsew', padx=8)

        self.txt_input = tk.Text(in_frame, bg=C['card'], fg=C['text1'],
                                 font=('Segoe UI', 12), bd=0, padx=14, pady=12,
                                 insertbackground=C['accent'], wrap='word',
                                 selectbackground=C['accent'],
                                 highlightthickness=0)
        sb_in = ttk.Scrollbar(in_frame, orient='vertical', command=self.txt_input.yview)
        self.txt_input.configure(yscrollcommand=sb_in.set)
        self.txt_input.pack(side='left', fill='both', expand=True)
        sb_in.pack(side='right', fill='y')

        # ── Right: Output ────────────────────────────────────────────────────
        lbl_out = tk.Label(body, text='MORSE CODE OUTPUT', bg=C['bg'],
                           fg=C['text3'], font=('Segoe UI', 9, 'bold'))
        lbl_out.grid(row=0, column=1, sticky='w', padx=8, pady=(0, 4))

        out_frame = tk.Frame(body, bg=C['input_bg'],
                             highlightbackground=C['border'], highlightthickness=1)
        out_frame.grid(row=1, column=1, sticky='nsew', padx=8)

        self.txt_output = tk.Text(out_frame, bg=C['input_bg'], fg=C['success'],
                                  font=('Consolas', 13), bd=0, padx=14, pady=12,
                                  state='disabled', wrap='word',
                                  selectbackground=C['accent'],
                                  highlightthickness=0)
        sb_out = ttk.Scrollbar(out_frame, orient='vertical', command=self.txt_output.yview)
        self.txt_output.configure(yscrollcommand=sb_out.set)
        self.txt_output.pack(side='left', fill='both', expand=True)
        sb_out.pack(side='right', fill='y')

        # ── Controls ─────────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=C['bg'])
        ctrl.pack(fill='x', padx=28, pady=(8, 20))

        _btn(ctrl, '   CONVERT   ', self._convert, C['accent']).pack(side='left', padx=(0, 8))
        _btn(ctrl, '   COPY   ', self._copy,
             '#21262D', C['text1']).pack(side='left', padx=(0, 8))
        _btn(ctrl, '   SAVE   ', self._save,
             '#21262D', C['text1']).pack(side='left', padx=(0, 8))
        _btn(ctrl, '   CLEAR   ', self._clear,
             C['error'], C['text1']).pack(side='left')

        self.status_var = tk.StringVar(value='Ready.')
        tk.Label(ctrl, textvariable=self.status_var, bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9)).pack(side='right')

        # Char count
        self.charcount_var = tk.StringVar(value='0 chars')
        tk.Label(ctrl, textvariable=self.charcount_var, bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9)).pack(side='right', padx=16)
        self.txt_input.bind('<KeyRelease>', self._on_key)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _on_key(self, event=None):
        n = len(self.txt_input.get('1.0', 'end').strip())
        self.charcount_var.set(f'{n} chars')

    def _convert(self):
        text = self.txt_input.get('1.0', 'end').strip()
        if not text:
            self.status_var.set('Error: Input is empty.')
            return
        morse = encode_text(text)
        self.txt_output.configure(state='normal')
        self.txt_output.delete('1.0', 'end')
        self.txt_output.insert('1.0', morse)
        self.txt_output.configure(state='disabled')
        self.status_var.set(f'Converted {len(text)} characters.')

    def _copy(self):
        morse = self.txt_output.get('1.0', 'end').strip()
        if not morse:
            self.status_var.set('Nothing to copy.')
            return
        self.clipboard_clear()
        self.clipboard_append(morse)
        self.status_var.set('Morse code copied to clipboard.')

    def _save(self):
        text = self.txt_input.get('1.0', 'end').strip()
        morse = self.txt_output.get('1.0', 'end').strip()
        if not morse:
            self.status_var.set('Nothing to save. Convert first.')
            return
        ok, msg = save_history({
            'direction':        'TX',
            'plain_text':       text,
            'morse_code':       morse,
            'encrypted_data':   '',
            'sender':           'LOCAL',
            'receiver':         'N/A',
            'encryption_status': 'None',
        })
        self.status_var.set(msg)

    def _clear(self):
        self.txt_input.delete('1.0', 'end')
        self.txt_output.configure(state='normal')
        self.txt_output.delete('1.0', 'end')
        self.txt_output.configure(state='disabled')
        self.status_var.set('Cleared.')
        self.charcount_var.set('0 chars')
