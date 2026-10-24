"""
LoRaShield – Message History Page
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from utils.history import load_history, delete_history_record, export_history, search_history


C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950', 'warning': '#D29922', 'error': '#F85149',
}

COLUMNS    = ('Time', 'Direction', 'Plain Text', 'Morse Code', 'Encrypted Data', 'Encryption')
COL_WIDTHS = (130, 70, 220, 180, 220, 100)


def _btn(parent, text, command, color=None, fg=None, **kw):
    color = color or C['accent']
    fg = fg or C['text1']
    b = tk.Button(parent, text=text, command=command, bg=color, fg=fg,
                  font=('Segoe UI', 9, 'bold'), bd=0, padx=12, pady=7,
                  cursor='hand2', activebackground=C['accent_h'],
                  activeforeground=C['text1'], relief='flat', **kw)
    b.bind('<Enter>', lambda e: b.config(bg=C['accent_h'] if color == C['accent'] else color))
    b.bind('<Leave>', lambda e: b.config(bg=color))
    return b


class HistoryPage(tk.Frame):
    """Message History page with searchable table."""

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._all_records: list[dict] = []
        self._build()
        self.refresh()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        tk.Label(hdr, text='MESSAGE HISTORY', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=28, pady=(10, 0))

        # ── Toolbar ─────────────────────────────────────────────────────────
        toolbar = tk.Frame(self, bg=C['bg'])
        toolbar.pack(fill='x', padx=28, pady=(12, 8))

        # Search
        tk.Label(toolbar, text='SEARCH', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        search_frame = tk.Frame(toolbar, bg=C['card'],
                                highlightbackground=C['border'], highlightthickness=1)
        search_frame.pack(side='left', padx=(8, 16))

        self.search_var = tk.StringVar()
        self.search_var.trace_add('write', lambda *a: self._do_search())
        tk.Entry(search_frame, textvariable=self.search_var, bg=C['card'],
                 fg=C['text1'], font=('Segoe UI', 10), bd=0, width=28,
                 insertbackground=C['accent'], highlightthickness=0
                 ).pack(padx=10, pady=6)

        _btn(toolbar, '  REFRESH  ', self.refresh, C['accent']).pack(side='left', padx=(0, 8))
        _btn(toolbar, '  EXPORT CSV  ', self._export, '#21262D', C['text1']).pack(side='left', padx=(0, 8))
        _btn(toolbar, '  DELETE ROW  ', self._delete_selected, C['error']).pack(side='left')

        self.count_var = tk.StringVar(value='0 records')
        tk.Label(toolbar, textvariable=self.count_var, bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9)).pack(side='right')

        # ── Table ────────────────────────────────────────────────────────────
        table_frame = tk.Frame(self, bg=C['card'],
                               highlightbackground=C['border'], highlightthickness=1)
        table_frame.pack(fill='both', expand=True, padx=28, pady=(0, 20))

        # Style the treeview
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('LoRa.Treeview',
                         background=C['card'],
                         foreground=C['text1'],
                         fieldbackground=C['card'],
                         rowheight=32,
                         font=('Segoe UI', 9),
                         borderwidth=0)
        style.configure('LoRa.Treeview.Heading',
                         background=C['card2'],
                         foreground=C['text2'],
                         font=('Segoe UI', 9, 'bold'),
                         relief='flat')
        style.map('LoRa.Treeview',
                   background=[('selected', C['accent'])],
                   foreground=[('selected', C['text1'])])
        style.map('LoRa.Treeview.Heading',
                   background=[('active', C['accent'])])

        self.tree = ttk.Treeview(table_frame, columns=COLUMNS,
                                  show='headings', style='LoRa.Treeview',
                                  selectmode='browse')
        for col, width in zip(COLUMNS, COL_WIDTHS):
            self.tree.heading(col, text=col,
                              command=lambda c=col: self._sort_by(c))
            self.tree.column(col, width=width, minwidth=60, anchor='w')

        # Alternating row colors
        self.tree.tag_configure('odd', background=C['card'])
        self.tree.tag_configure('even', background=C['card2'])

        sb_v = ttk.Scrollbar(table_frame, orient='vertical', command=self.tree.yview)
        sb_h = ttk.Scrollbar(table_frame, orient='horizontal', command=self.tree.xview)
        self.tree.configure(yscrollcommand=sb_v.set, xscrollcommand=sb_h.set)

        sb_v.pack(side='right', fill='y')
        sb_h.pack(side='bottom', fill='x')
        self.tree.pack(fill='both', expand=True)

        # Status bar
        self.status_var = tk.StringVar(value='')
        tk.Label(self, textvariable=self.status_var, bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 8)).pack(anchor='w', padx=28, pady=(0, 4))

    # ── Data Methods ──────────────────────────────────────────────────────────

    def refresh(self):
        """Reload all records from disk."""
        self._all_records = load_history()
        self._populate(self._all_records)
        self.status_var.set('Records loaded from disk.')

    def _populate(self, records: list[dict]):
        self.tree.delete(*self.tree.get_children())
        for i, rec in enumerate(records):
            tag = 'even' if i % 2 == 0 else 'odd'
            self.tree.insert('', 'end', iid=str(i), tags=(tag,), values=(
                rec.get('time', ''),
                rec.get('direction', ''),
                rec.get('plain_text', rec.get('original_text', ''))[:40],
                rec.get('morse_code', '')[:40],
                rec.get('encrypted_data', '')[:40],
                rec.get('encryption_status', ''),
            ))
        self.count_var.set(f'{len(records)} records')

    def _do_search(self):
        q = self.search_var.get().strip()
        if not q:
            self._populate(self._all_records)
        else:
            results = search_history(q)
            self._populate(results)

    def _sort_by(self, col: str):
        col_map = {
            'Time':           'time',
            'Direction':      'direction',
            'Plain Text':     'plain_text',
            'Morse Code':     'morse_code',
            'Encrypted Data': 'encrypted_data',
            'Encryption':     'encryption_status',
        }
        key = col_map.get(col, 'time')
        self._all_records.sort(key=lambda r: str(r.get(key, '')).lower())
        self._populate(self._all_records)

    def _delete_selected(self):
        sel = self.tree.selection()
        if not sel:
            self.status_var.set('No row selected.')
            return
        idx = int(sel[0])
        if messagebox.askyesno('Confirm Delete',
                                'Delete this message record?'):
            ok, msg = delete_history_record(idx)
            self.status_var.set(msg)
            self.refresh()

    def _export(self):
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Export Message History',
        )
        if path:
            ok, msg = export_history(path)
            self.status_var.set(msg)
