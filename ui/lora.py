"""
LoRaShield – LoRa Communication Page
"""

import tkinter as tk
from tkinter import ttk
import time
import threading
from utils import serial_comm
from utils.encryption import encrypt_message, decrypt_message
from utils.history import save_history
from utils.crc import build_packet, parse_packet


C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950', 'warning': '#D29922', 'error': '#F85149',
    'terminal': '#010409', 'term_txt': '#39D353',
}

BAUD_RATES = ['9600', '19200', '38400', '57600', '115200', '230400']


def _btn(parent, text, command, color=None, fg=None, state='normal', **kw):
    color = color or C['accent']
    fg = fg or C['text1']
    b = tk.Button(parent, text=text, command=command, bg=color, fg=fg,
                  font=('Segoe UI', 10, 'bold'), bd=0, padx=14, pady=8,
                  cursor='hand2', activebackground=C['accent_h'],
                  activeforeground=C['text1'], relief='flat', state=state, **kw)
    b.bind('<Enter>', lambda e: b.config(bg=C['accent_h'] if color == C['accent'] else color))
    b.bind('<Leave>', lambda e: b.config(bg=color))
    return b


class LoRaPage(tk.Frame):
    """LoRa Serial Communication page."""

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._connected = False
        self._build()
        self._refresh_ports()

    def _build(self):
        # Header
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        tk.Label(hdr, text='LORA COMMUNICATION', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')

        # Connection LED
        self.led_canvas = tk.Canvas(hdr, width=14, height=14,
                                    bg=C['bg'], bd=0, highlightthickness=0)
        self.led_canvas.pack(side='left', padx=(16, 4))
        self._led = self.led_canvas.create_oval(2, 2, 12, 12,
                                                fill=C['error'], outline='')
        self.conn_label = tk.Label(hdr, text='Disconnected', bg=C['bg'],
                                   fg=C['error'], font=('Segoe UI', 9, 'bold'))
        self.conn_label.pack(side='left')

        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=28, pady=(10, 0))

        # ── Config bar ───────────────────────────────────────────────────────
        cfg = tk.Frame(self, bg=C['card2'],
                       highlightbackground=C['border'], highlightthickness=1)
        cfg.pack(fill='x', padx=28, pady=(14, 0))

        ci = tk.Frame(cfg, bg=C['card2'])
        ci.pack(fill='x', padx=16, pady=12)

        # COM Port
        tk.Label(ci, text='COM PORT', bg=C['card2'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(ci, textvariable=self.port_var,
                                       width=12, state='readonly', style='TCombobox')
        self.port_combo.pack(side='left', padx=(8, 20))

        tk.Button(ci, text='REFRESH', command=self._refresh_ports,
                  bg=C['card2'], fg=C['accent'], font=('Segoe UI', 8, 'bold'),
                  bd=0, padx=8, pady=4, cursor='hand2', relief='flat'
                  ).pack(side='left', padx=(0, 20))

        # Baud Rate
        tk.Label(ci, text='BAUD RATE', bg=C['card2'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')
        self.baud_var = tk.StringVar(value='9600')
        ttk.Combobox(ci, textvariable=self.baud_var,
                     values=BAUD_RATES, width=10, state='readonly'
                     ).pack(side='left', padx=(8, 20))

        # Connect / Disconnect
        self.btn_connect = _btn(ci, '  CONNECT  ', self._connect, C['success'])
        self.btn_connect.pack(side='left', padx=(0, 8))

        self.btn_disconnect = _btn(ci, '  DISCONNECT  ', self._disconnect,
                                   C['error'], state='disabled')
        self.btn_disconnect.pack(side='left')

        # ── Main body ────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=20, pady=14)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Terminal
        term_frame = tk.Frame(body, bg=C['terminal'],
                              highlightbackground=C['border'], highlightthickness=1)
        term_frame.grid(row=0, column=0, sticky='nsew', padx=(8, 4))

        term_hdr = tk.Frame(term_frame, bg='#0A0A0A')
        term_hdr.pack(fill='x')
        tk.Label(term_hdr, text='  SERIAL TERMINAL', bg='#0A0A0A',
                 fg=C['term_txt'], font=('Consolas', 9)).pack(side='left', pady=6)
        tk.Button(term_hdr, text='CLEAR', command=self._clear_terminal,
                  bg='#0A0A0A', fg=C['text3'], font=('Segoe UI', 8),
                  bd=0, padx=8, cursor='hand2', relief='flat').pack(side='right')

        self.terminal = tk.Text(term_frame, bg=C['terminal'], fg=C['term_txt'],
                                font=('Consolas', 10), bd=0, padx=12, pady=8,
                                state='disabled', wrap='word',
                                selectbackground=C['accent'],
                                highlightthickness=0)
        sb_t = ttk.Scrollbar(term_frame, orient='vertical', command=self.terminal.yview)
        self.terminal.configure(yscrollcommand=sb_t.set)
        self.terminal.pack(side='left', fill='both', expand=True)
        sb_t.pack(side='right', fill='y')

        # Connection log
        log_frame = tk.Frame(body, bg=C['card'],
                             highlightbackground=C['border'], highlightthickness=1)
        log_frame.grid(row=0, column=1, sticky='nsew', padx=(4, 8))

        tk.Label(log_frame, text='CONNECTION LOG', bg=C['card'],
                 fg=C['text3'], font=('Segoe UI', 8, 'bold'), pady=6).pack(fill='x', padx=10)
        tk.Frame(log_frame, bg=C['border'], height=1).pack(fill='x')

        self.log_box = tk.Text(log_frame, bg=C['card'], fg=C['text2'],
                               font=('Consolas', 8), bd=0, padx=10, pady=6,
                               state='disabled', wrap='word',
                               highlightthickness=0)
        sb_l = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=sb_l.set)
        self.log_box.pack(side='left', fill='both', expand=True)
        sb_l.pack(side='right', fill='y')

        # ── Send row ─────────────────────────────────────────────────────────
        send_row = tk.Frame(self, bg=C['card2'],
                            highlightbackground=C['border'], highlightthickness=1)
        send_row.pack(fill='x', padx=28, pady=(0, 8))

        sr = tk.Frame(send_row, bg=C['card2'])
        sr.pack(fill='x', padx=12, pady=10)

        self.send_var = tk.StringVar()
        send_entry = tk.Entry(sr, textvariable=self.send_var, bg=C['bg'],
                              fg=C['text1'], font=('Segoe UI', 11), bd=0,
                              insertbackground=C['accent'],
                              selectbackground=C['accent'],
                              highlightthickness=0)
        send_entry.pack(side='left', fill='x', expand=True, padx=(0, 12), ipady=6)
        send_entry.bind('<Return>', lambda e: self._send())

        _btn(sr, '  SEND  ', self._send, C['accent']).pack(side='left', padx=(0, 8))
        _btn(sr, '  RECEIVE  ', self._receive, '#21262D', C['text1']).pack(side='left')

        self._term_log('LoRaShield serial terminal ready.')
        self._term_log('Select COM port and baud rate, then press CONNECT.')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _term_log(self, msg: str, tag: str = 'info'):
        ts = time.strftime('%H:%M:%S')
        self.terminal.configure(state='normal')
        self.terminal.insert('end', f'[{ts}] {msg}\n')
        self.terminal.see('end')
        self.terminal.configure(state='disabled')

    def _conn_log(self, msg: str):
        ts = time.strftime('%H:%M:%S')
        self.log_box.configure(state='normal')
        self.log_box.insert('end', f'[{ts}] {msg}\n')
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def _set_connected(self, connected: bool):
        self._connected = connected
        if connected:
            self.led_canvas.itemconfig(self._led, fill=C['success'])
            self.conn_label.config(text='Connected', fg=C['success'])
            self.btn_connect.config(state='disabled')
            self.btn_disconnect.config(state='normal')
        else:
            self.led_canvas.itemconfig(self._led, fill=C['error'])
            self.conn_label.config(text='Disconnected', fg=C['error'])
            self.btn_connect.config(state='normal')
            self.btn_disconnect.config(state='disabled')

    # ── Actions ───────────────────────────────────────────────────────────────

    def _refresh_ports(self):
        ports = serial_comm.list_com_ports()
        self.port_combo['values'] = ports
        if ports:
            self.port_combo.current(0)

    def _connect(self):
        port = self.port_var.get()
        baud = int(self.baud_var.get())
        ok, msg = serial_comm.connect_esp32(port, baud, self._conn_log)
        self._term_log(msg)
        if ok:
            self._set_connected(True)
            self.app_state['com_port'] = port
            def _on_rx(data: str) -> None:
                # ── CRC verification ──────────────────────────────────────────
                parsed = parse_packet(data)
                if not parsed['valid']:
                    # Corrupted packet – log and reject without decryption
                    self.after(0, lambda e=parsed['error']: self._term_log(
                        f'[CRC FAILED] PACKET CORRUPTED – {e}'))
                    self.after(0, lambda: self._term_log(
                        '[CRC: FAILED - PACKET CORRUPTED]'))
                    return

                if parsed['is_legacy']:
                    self.after(0, lambda: self._term_log('[CRC] Legacy packet (no CRC field).'))
                else:
                    self.after(0, lambda c=parsed['crc_rx']: self._term_log(
                        f'[CRC: VALID] 0x{c:04X}'))

                payload = parsed['payload']

                key = self.app_state.get('enc_key', '')
                if key and ':' in payload:
                    plaintext, dec_err = decrypt_message(payload, key)
                    if dec_err:
                        self.after(0, lambda d=payload: self._term_log(f'[RX CIPHER] {d}'))
                        self.after(0, lambda e=dec_err: self._term_log(f'[RX DECRYPT ERR] {e}'))
                        plain_display = payload
                        enc_status = 'Decrypt Error'
                        save_enc = ''
                    else:
                        self.after(0, lambda d=payload: self._term_log(f'[RX CIPHER] {d}'))
                        self.after(0, lambda p=plaintext: self._term_log(f'[RX PLAIN ] {p}'))
                        plain_display = plaintext
                        enc_status = 'AES-256 CBC'
                        save_enc = payload
                else:
                    self.after(0, lambda d=payload: self._term_log(f'[RX] {d}'))
                    plain_display = payload
                    enc_status = 'None'
                    save_enc = ''

                save_history({
                    'direction':        'RX',
                    'plain_text':       plain_display,
                    'morse_code':       '',
                    'encrypted_data':   save_enc,
                    'sender':           self.app_state.get('com_port', 'Remote'),
                    'receiver':         'LOCAL',
                    'encryption_status': enc_status,
                })

            serial_comm.start_receive_loop(
                callback=_on_rx,
                error_callback=lambda err: self.after(0, lambda: self._term_log(f'[ERR] {err}')),
            )

    def _disconnect(self):
        serial_comm.stop_receive_loop()
        ok, msg = serial_comm.disconnect_esp32(self._conn_log)
        self._term_log(msg)
        if ok:
            self._set_connected(False)

    def _send(self):
        msg = self.send_var.get().strip()
        if not msg:
            return

        key = self.app_state.get('enc_key', '')

        # ── Encrypt the message before sending ───────────────────────────────
        if key:
            encrypted, err = encrypt_message(msg, key)
            if err:
                self._term_log(f'[ENCRYPT ERR] {err}')
                return
            payload = encrypted
            self._term_log(f'[TX PLAIN ] {msg}')
            self._term_log(f'[TX CIPHER] {payload}')
        else:
            payload = msg
            encrypted = ''
            self._term_log(f'[TX] {msg} (no key – sent unencrypted)')

        # ── Wrap in CRC packet ───────────────────────────────────────────────
        packet = build_packet(payload)
        from utils.crc import calculate_crc as _crc
        _crc_val = _crc(payload)
        self._term_log(f'[CRC GENERATED] 0x{_crc_val:04X} ({_crc_val})')
        self._term_log(f'[TX PACKET] {packet}')

        ok, result = serial_comm.send_lora(packet, self._conn_log)
        if ok:
            self._term_log('[PACKET SENT]')
            self.send_var.set('')
            # Save to history
            save_history({
                'direction':        'TX',
                'plain_text':       msg,
                'morse_code':       '',
                'encrypted_data':   encrypted,
                'sender':           'LOCAL',
                'receiver':         self.app_state.get('com_port', 'Remote'),
                'encryption_status': 'AES-256 CBC' if key else 'None',
            })
        else:
            self._term_log(f'[ERR] {result}')

    def _receive(self):
        data, err = serial_comm.receive_lora(timeout=3.0)
        if data:
            # ── CRC verification ─────────────────────────────────────────────
            parsed = parse_packet(data)
            if not parsed['valid']:
                self._term_log(f'[CRC FAILED] PACKET CORRUPTED – {parsed["error"]}')
                self._term_log('[CRC: FAILED - PACKET CORRUPTED]')
                return

            if parsed['is_legacy']:
                self._term_log('[CRC] Legacy packet (no CRC field).')
            else:
                self._term_log(f'[CRC: VALID] 0x{parsed["crc_rx"]:04X}')

            payload = parsed['payload']
            key = self.app_state.get('enc_key', '')

            # Attempt decryption if we have a key
            if key and ':' in payload:
                plaintext, dec_err = decrypt_message(payload, key)
                if dec_err:
                    self._term_log(f'[RX CIPHER] {payload}')
                    self._term_log(f'[RX DECRYPT ERR] {dec_err}')
                    plain_display = payload
                    enc_status = 'Decrypt Error'
                else:
                    self._term_log(f'[RX CIPHER] {payload}')
                    self._term_log(f'[RX PLAIN ] {plaintext}')
                    plain_display = plaintext
                    enc_status = 'AES-256 CBC'
            else:
                self._term_log(f'[RX] {payload}')
                plain_display = payload
                enc_status = 'None'
                key = ''

            # Save received message to history
            save_history({
                'direction':        'RX',
                'plain_text':       plain_display,
                'morse_code':       '',
                'encrypted_data':   payload if key else '',
                'sender':           self.app_state.get('com_port', 'Remote'),
                'receiver':         'LOCAL',
                'encryption_status': enc_status,
            })
        else:
            self._term_log(f'[RX] {err}')

    def _clear_terminal(self):
        self.terminal.configure(state='normal')
        self.terminal.delete('1.0', 'end')
        self.terminal.configure(state='disabled')
