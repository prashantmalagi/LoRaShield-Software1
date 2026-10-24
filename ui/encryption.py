"""
LoRaShield – AES Encryption Page
"""

import tkinter as tk
from tkinter import ttk
from utils.encryption import encrypt_message, decrypt_message, is_crypto_available, generate_key


C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950', 'warning': '#D29922', 'error': '#F85149',
    'teal': '#14B8A6', 'input_bg': '#0D1117',
}


def _btn(parent, text, command, color=None, fg=None, **kw):
    color = color or C['accent']
    fg = fg or C['text1']
    kw.setdefault('padx', 16)
    kw.setdefault('pady', 8)
    kw.setdefault('font', ('Segoe UI', 10, 'bold'))
    b = tk.Button(parent, text=text, command=command, bg=color, fg=fg,
                  bd=0, cursor='hand2', activebackground=C['accent_h'],
                  activeforeground=C['text1'], relief='flat', **kw)
    b.bind('<Enter>', lambda e: b.config(bg=C['accent_h'] if color == C['accent'] else color))
    b.bind('<Leave>', lambda e: b.config(bg=color))
    return b


def _section(parent, title: str) -> tk.Frame:
    tk.Label(parent, text=title, bg=C['bg'],
             fg=C['text3'], font=('Segoe UI', 9, 'bold')
             ).pack(anchor='w', padx=0, pady=(0, 4))
    f = tk.Frame(parent, bg=C['card'],
                 highlightbackground=C['border'], highlightthickness=1)
    f.pack(fill='x')
    return f


class EncryptionPage(tk.Frame):
    """AES-256 Encryption / Decryption page."""

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._build()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        tk.Label(hdr, text='AES-256 ENCRYPTION', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')

        crypto_ok = is_crypto_available()
        status_txt = 'pycryptodome Ready' if crypto_ok else 'pycryptodome Not Installed'
        status_col = C['success'] if crypto_ok else C['error']
        tk.Label(hdr, text=f'  [{status_txt}]', bg=C['bg'],
                 fg=status_col, font=('Segoe UI', 9)).pack(side='left', padx=10)

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

        # Bind mousewheel
        def _mw(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _mw)

        pad = {'padx': 28, 'pady': 8}

        # ── Secret Key ────────────────────────────────────────────────────────
        key_sect = tk.Frame(body, bg=C['bg'])
        key_sect.pack(fill='x', **pad)

        tk.Label(key_sect, text='SECRET KEY', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(anchor='w')

        key_frame = tk.Frame(key_sect, bg=C['card'],
                             highlightbackground=C['border'], highlightthickness=1)
        key_frame.pack(fill='x', pady=(4, 0))

        self.key_var = tk.StringVar(value=self.app_state.get('enc_key', ''))
        self.key_entry = tk.Entry(key_frame, textvariable=self.key_var,
                                  show='*', bg=C['card'], fg=C['text1'],
                                  font=('Consolas', 11), bd=0, insertbackground=C['accent'],
                                  selectbackground=C['accent'],
                                  highlightthickness=0)
        self.key_entry.pack(side='left', fill='x', expand=True, padx=14, pady=10)

        self.show_key = False

        def toggle_key():
            self.show_key = not self.show_key
            self.key_entry.config(show='' if self.show_key else '*')
            toggle_btn.config(text='HIDE' if self.show_key else 'SHOW')

        toggle_btn = tk.Button(key_frame, text='SHOW', command=toggle_key,
                               bg=C['card'], fg=C['accent'], font=('Segoe UI', 9, 'bold'),
                               bd=0, padx=12, cursor='hand2', relief='flat')
        toggle_btn.pack(side='right', padx=8)

        # ── Generate Key / Copy Key buttons ──────────────────────────────────
        key_btn_row = tk.Frame(key_sect, bg=C['bg'])
        key_btn_row.pack(fill='x', pady=(6, 0))

        _btn(key_btn_row, '  GENERATE KEY  ', self._generate_key,
             C['teal']).pack(side='left', padx=(0, 8))
        _btn(key_btn_row, '  COPY KEY  ', self._copy_key,
             '#21262D', C['text1']).pack(side='left')

        self.key_status_var = tk.StringVar(value='')
        tk.Label(key_btn_row, textvariable=self.key_status_var, bg=C['bg'],
                 fg=C['success'], font=('Segoe UI', 9)).pack(side='left', padx=12)

        tk.Label(key_sect,
                 text='Key is SHA-256 hashed to 256 bits internally. Any length accepted.',
                 bg=C['bg'], fg=C['text3'], font=('Segoe UI', 8)).pack(anchor='w', pady=(4, 0))

        # ── Input ─────────────────────────────────────────────────────────────
        in_sect = tk.Frame(body, bg=C['bg'])
        in_sect.pack(fill='x', **pad)

        tk.Label(in_sect, text='PLAINTEXT INPUT', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(anchor='w')

        in_frame = tk.Frame(in_sect, bg=C['card'],
                            highlightbackground=C['border'], highlightthickness=1)
        in_frame.pack(fill='x', pady=(4, 0))

        self.txt_input = tk.Text(in_frame, bg=C['card'], fg=C['text1'],
                                 font=('Segoe UI', 11), bd=0, padx=14, pady=10,
                                 height=5, insertbackground=C['accent'],
                                 selectbackground=C['accent'],
                                 highlightthickness=0, wrap='word')
        self.txt_input.pack(fill='x')

        # ── Buttons row 1 ─────────────────────────────────────────────────────
        btn_row1 = tk.Frame(body, bg=C['bg'])
        btn_row1.pack(fill='x', padx=28, pady=(0, 4))
        _btn(btn_row1, '  ENCRYPT  ', self._encrypt, C['accent']).pack(side='left', padx=(0, 10))
        _btn(btn_row1, '  DECRYPT  ', self._decrypt, C['teal']).pack(side='left', padx=(0, 10))
        _btn(btn_row1, '  CLEAR ALL  ', self._clear_all, C['error']).pack(side='left')

        self.status_var = tk.StringVar(value='Ready.')
        tk.Label(btn_row1, textvariable=self.status_var, bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9)).pack(side='right')

        # ── Encrypted Output ──────────────────────────────────────────────────
        enc_sect = tk.Frame(body, bg=C['bg'])
        enc_sect.pack(fill='x', **pad)

        enc_hdr = tk.Frame(enc_sect, bg=C['bg'])
        enc_hdr.pack(fill='x')
        tk.Label(enc_hdr, text='ENCRYPTED OUTPUT (AES-256 CBC / Base64)',
                 bg=C['bg'], fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        _btn(enc_hdr, 'COPY', self._copy_enc, '#21262D', C['text1'],
             padx=10, pady=4, font=('Segoe UI', 8, 'bold')).pack(side='right')

        enc_frame = tk.Frame(enc_sect, bg=C['input_bg'],
                             highlightbackground=C['border'], highlightthickness=1)
        enc_frame.pack(fill='x', pady=(4, 0))

        self.txt_encrypted = tk.Text(enc_frame, bg=C['input_bg'], fg=C['warning'],
                                     font=('Consolas', 10), bd=0, padx=14, pady=10,
                                     height=4, state='disabled', wrap='word',
                                     selectbackground=C['accent'],
                                     highlightthickness=0)
        self.txt_encrypted.pack(fill='x')

        # ── Decrypted Output ──────────────────────────────────────────────────
        dec_sect = tk.Frame(body, bg=C['bg'])
        dec_sect.pack(fill='x', **pad)

        dec_hdr = tk.Frame(dec_sect, bg=C['bg'])
        dec_hdr.pack(fill='x')
        tk.Label(dec_hdr, text='DECRYPTED OUTPUT', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        _btn(dec_hdr, 'COPY', self._copy_dec, '#21262D', C['text1'],
             padx=10, pady=4, font=('Segoe UI', 8, 'bold')).pack(side='right')

        dec_frame = tk.Frame(dec_sect, bg=C['card'],
                             highlightbackground=C['border'], highlightthickness=1)
        dec_frame.pack(fill='x', pady=(4, 0))

        self.txt_decrypted = tk.Text(dec_frame, bg=C['card'], fg=C['success'],
                                     font=('Segoe UI', 11), bd=0, padx=14, pady=10,
                                     height=4, state='disabled', wrap='word',
                                     selectbackground=C['accent'],
                                     highlightthickness=0)
        self.txt_decrypted.pack(fill='x')

        # ── Decrypt from ciphertext field ─────────────────────────────────────
        dec2_sect = tk.Frame(body, bg=C['bg'])
        dec2_sect.pack(fill='x', padx=28, pady=(0, 4))
        tk.Label(dec2_sect,
                 text='To decrypt: paste ciphertext in "Plaintext Input" field and press DECRYPT.',
                 bg=C['bg'], fg=C['text3'], font=('Segoe UI', 8)).pack(anchor='w')

        tk.Frame(body, bg=C['bg'], height=20).pack()

    # ── Key Actions ──────────────────────────────────────────────────────────

    def _generate_key(self):
        new_key = generate_key(32)
        self.key_var.set(new_key)
        # Show the key so user can inspect it
        self.show_key = True
        self.key_entry.config(show='')
        self.key_status_var.set('New key generated.')
        self.status_var.set('New 32-character key generated. Save it securely.')

    def _copy_key(self):
        key = self.key_var.get()
        if key:
            self.clipboard_clear()
            self.clipboard_append(key)
            self.key_status_var.set('Key copied to clipboard.')
        else:
            self.key_status_var.set('No key to copy.')

    # ── Encrypt / Decrypt Actions ─────────────────────────────────────────────

    def _get_key(self) -> str | None:
        key = self.key_var.get().strip()
        if not key:
            self.status_var.set('Error: Secret key is required.')
            return None
        return key

    def _set_output(self, widget: tk.Text, text: str):
        widget.configure(state='normal')
        widget.delete('1.0', 'end')
        widget.insert('1.0', text)
        widget.configure(state='disabled')

    def _encrypt(self):
        key = self._get_key()
        if key is None:
            return
        msg = self.txt_input.get('1.0', 'end').strip()
        if not msg:
            self.status_var.set('Error: Plaintext input is empty.')
            return
        enc, err = encrypt_message(msg, key)
        if err:
            self.status_var.set(f'Error: {err}')
        else:
            self._set_output(self.txt_encrypted, enc)
            self.status_var.set('Encrypted successfully with AES-256 CBC.')

    def _decrypt(self):
        key = self._get_key()
        if key is None:
            return
        cipher_text = self.txt_input.get('1.0', 'end').strip()
        if not cipher_text:
            self.status_var.set('Error: Paste ciphertext in the input field to decrypt.')
            return
        dec, err = decrypt_message(cipher_text, key)
        if err:
            self.status_var.set(f'Error: {err}')
        else:
            self._set_output(self.txt_decrypted, dec)
            self.status_var.set('Decrypted successfully.')

    def _copy_enc(self):
        t = self.txt_encrypted.get('1.0', 'end').strip()
        if t:
            self.clipboard_clear()
            self.clipboard_append(t)
            self.status_var.set('Encrypted text copied.')

    def _copy_dec(self):
        t = self.txt_decrypted.get('1.0', 'end').strip()
        if t:
            self.clipboard_clear()
            self.clipboard_append(t)
            self.status_var.set('Decrypted text copied.')

    def _clear_all(self):
        self.txt_input.delete('1.0', 'end')
        self._set_output(self.txt_encrypted, '')
        self._set_output(self.txt_decrypted, '')
        self.status_var.set('Cleared.')
