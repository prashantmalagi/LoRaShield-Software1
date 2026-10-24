"""
LoRaShield – Morse to Text (AI) Decoder Page
=============================================
Extended with an interactive Morse Code Typing Error Correction dialog.

Architecture:
    MANUAL USER INPUT:
        Morse Dictionary + Similarity Matching  (utils.morse_correction)
            ↓ Correction dialog
            ↓ User selects suggestion
        Normal decode (utils.ai.decode_morse)

    NOISY / UNCERTAIN MORSE:
        Existing AI model (morse_decoder_v4.keras)
            ↓ AI prediction

The AI model, existing decode flow, and all original UI elements are
completely preserved.  The correction feature is purely additive.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading

from utils.ai import load_ai_model, decode_morse, is_model_loaded
from utils.morse_correction import (
    correct_morse_message,
    apply_correction,
    _DEFAULT_THRESHOLD,
    set_default_threshold,
)
from utils.logger import get_logger

log = get_logger(__name__)


C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950', 'warning': '#D29922', 'error': '#F85149',
    'purple': '#8B5CF6', 'input_bg': '#0D1117',
}


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


# ─────────────────────────────────────────────────────────────────────────────
# Morse Correction Dialog
# ─────────────────────────────────────────────────────────────────────────────

class MorseCorrectionDialog(tk.Toplevel):
    """
    Modal dialog that shows typing-error suggestions for invalid Morse tokens.

    For each invalid token the dialog displays:
        • The invalid pattern in red / orange.
        • A "Did you mean?" label.
        • Up to 3 clickable suggestion buttons.
        • "No reliable correction found." for below-threshold tokens.

    When the user clicks a suggestion the token is replaced in the parent
    input field.  The dialog re-checks the (now corrected) message and
    refreshes its display.  An Apply button closes the dialog after all
    corrections are made.

    Parameters
    ----------
    parent_page : DecoderPage
        The DecoderPage instance that owns this dialog.
    morse_message : str
        The current Morse input string from txt_input.
    threshold : float
        Similarity threshold passed to the correction engine.
    """

    def __init__(self, parent_page, morse_message: str, threshold: float):
        super().__init__(parent_page, bg=C['bg'])
        self.parent_page  = parent_page
        self.threshold    = threshold
        self._morse       = morse_message
        self._result      = None          # Final (corrected) Morse string
        self._token_btns  = []            # Track dynamically created widgets

        self.title('Morse Code Correction')
        self.resizable(True, True)
        self.minsize(560, 400)
        self.grab_set()                   # Modal: block interaction with parent

        self._build_static()
        self._run_check()
        self.wait_window(self)            # Block until dialog closes

    # ─── Static frame layout ──────────────────────────────────────────────

    def _build_static(self):
        # ── Header ──────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=20, pady=(16, 0))

        tk.Label(hdr, text='⚡  MORSE CODE CORRECTION',
                 bg=C['bg'], fg=C['warning'],
                 font=('Segoe UI', 11, 'bold')).pack(side='left')
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=20, pady=(8, 0))

        # ── Current Morse display ────────────────────────────────────────
        cur_frame = tk.Frame(self, bg=C['card2'],
                             highlightbackground=C['border'], highlightthickness=1)
        cur_frame.pack(fill='x', padx=20, pady=(10, 0))

        inner = tk.Frame(cur_frame, bg=C['card2'])
        inner.pack(fill='x', padx=12, pady=8)

        tk.Label(inner, text='Input:', bg=C['card2'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        self.cur_label = tk.Label(inner, text=self._morse, bg=C['card2'],
                                  fg=C['success'], font=('Consolas', 12),
                                  wraplength=480, justify='left')
        self.cur_label.pack(side='left', padx=(8, 0))

        # ── Threshold control ────────────────────────────────────────────
        thr_frame = tk.Frame(self, bg=C['bg'])
        thr_frame.pack(fill='x', padx=20, pady=(8, 0))

        tk.Label(thr_frame, text='Similarity threshold:',
                 bg=C['bg'], fg=C['text3'], font=('Segoe UI', 9)).pack(side='left')

        self.thr_var = tk.DoubleVar(value=round(self.threshold * 100))
        self.thr_label = tk.Label(thr_frame,
                                  text=f'{int(self.thr_var.get())} %',
                                  bg=C['bg'], fg=C['accent'],
                                  font=('Consolas', 10, 'bold'), width=5)
        self.thr_label.pack(side='right', padx=(0, 8))

        thr_scale = tk.Scale(thr_frame, from_=50, to=95, orient='horizontal',
                             variable=self.thr_var, bg=C['bg'], fg=C['text2'],
                             highlightthickness=0, troughcolor=C['card2'],
                             activebackground=C['accent'], showvalue=False,
                             command=self._on_threshold_change)
        thr_scale.pack(side='left', fill='x', expand=True, padx=(8, 8))

        # ── Scrollable corrections area ──────────────────────────────────
        tk.Label(self, text='CORRECTIONS', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')
                 ).pack(anchor='w', padx=20, pady=(14, 4))

        list_outer = tk.Frame(self, bg=C['card'],
                              highlightbackground=C['border'], highlightthickness=1)
        list_outer.pack(fill='both', expand=True, padx=20, pady=(0, 8))

        canvas = tk.Canvas(list_outer, bg=C['card'], bd=0, highlightthickness=0)
        sb     = ttk.Scrollbar(list_outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        self._scroll_frame = tk.Frame(canvas, bg=C['card'])
        self._scroll_win   = canvas.create_window((0, 0), window=self._scroll_frame,
                                                  anchor='nw')
        self._scroll_frame.bind('<Configure>',
                                lambda e: canvas.configure(
                                    scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfig(self._scroll_win, width=e.width))

        # ── Status label ─────────────────────────────────────────────────
        self.status_var = tk.StringVar(value='Checking…')
        tk.Label(self, textvariable=self.status_var, bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9)
                 ).pack(anchor='w', padx=20, pady=(0, 4))

        # ── Bottom buttons ───────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=C['bg'])
        btn_row.pack(fill='x', padx=20, pady=(0, 16))

        _btn(btn_row, '  Apply & Close  ', self._apply_and_close,
             C['success']).pack(side='left')
        _btn(btn_row, '  Cancel  ', self.destroy,
             C['error']).pack(side='left', padx=(8, 0))

    # ─── Dynamic correction rows ──────────────────────────────────────────

    def _clear_rows(self):
        """Remove all dynamically created correction rows."""
        for w in self._scroll_frame.winfo_children():
            w.destroy()
        self._token_btns.clear()

    def _build_correction_rows(self, check_result: dict):
        """Build one row per invalid token inside the scrollable frame."""
        self._clear_rows()
        tokens = check_result.get('tokens', [])
        invalid = [t for t in tokens if not t['is_valid']]

        if not invalid:
            # All valid – show a success message
            ok_lbl = tk.Label(self._scroll_frame,
                              text='✓  All Morse patterns are valid!',
                              bg=C['card'], fg=C['success'],
                              font=('Segoe UI', 11, 'bold'))
            ok_lbl.pack(padx=20, pady=20)
            self.status_var.set(f'All {check_result["token_count"]} token(s) valid.')
            return

        self.status_var.set(
            f'{len(invalid)} invalid token(s) found. Click a suggestion to correct.'
        )

        for tok in invalid:
            self._build_token_row(tok)

    def _build_token_row(self, token: dict):
        """Build the UI row for one invalid Morse token."""
        word_idx  = token['word_index']
        tok_idx   = token['token_index']
        code      = token['input']
        reliable  = token['reliable']
        suggs     = token['suggestions']

        row_color = C['warning'] if reliable else C['error']

        # ── Outer frame ──────────────────────────────────────────────────
        row = tk.Frame(self._scroll_frame, bg=C['card2'],
                       highlightbackground=row_color, highlightthickness=1)
        row.pack(fill='x', padx=12, pady=6)

        # ── Invalid code label ───────────────────────────────────────────
        top = tk.Frame(row, bg=C['card2'])
        top.pack(fill='x', padx=10, pady=(8, 4))

        tk.Label(top, text=code, bg=C['card2'], fg=row_color,
                 font=('Consolas', 13, 'bold')).pack(side='left')

        if reliable:
            tk.Label(top, text='– Unknown Morse code.  Did you mean?',
                     bg=C['card2'], fg=C['text3'],
                     font=('Segoe UI', 9)).pack(side='left', padx=(8, 0))
        else:
            tk.Label(top, text='– No reliable Morse correction found.',
                     bg=C['card2'], fg=C['error'],
                     font=('Segoe UI', 9)).pack(side='left', padx=(8, 0))

        # ── Suggestion buttons ───────────────────────────────────────────
        if reliable and suggs:
            btn_row = tk.Frame(row, bg=C['card2'])
            btn_row.pack(fill='x', padx=10, pady=(0, 8))

            for s in suggs:
                char  = s['char']
                smorse = s['morse']
                score = s['score']
                pct   = int(score * 100)

                lbl_text = f'  {char}  ({smorse})  {pct}%  '
                b = tk.Button(
                    btn_row,
                    text=lbl_text,
                    command=lambda c=char, m=smorse, wi=word_idx, ti=tok_idx:
                        self._apply_suggestion(c, m, wi, ti),
                    bg=C['card'],
                    fg=C['accent'],
                    font=('Consolas', 10, 'bold'),
                    bd=1,
                    relief='solid',
                    padx=8, pady=4,
                    cursor='hand2',
                    activebackground=C['accent'],
                    activeforeground=C['text1'],
                )
                b.pack(side='left', padx=(0, 6))
                self._token_btns.append(b)
        elif not reliable:
            tk.Label(row,
                     text='  ⚠  Too different from any known Morse character. Please re-enter.',
                     bg=C['card2'], fg=C['error'],
                     font=('Segoe UI', 9)).pack(anchor='w', padx=10, pady=(0, 8))

    # ─── Event handlers ───────────────────────────────────────────────────

    def _on_threshold_change(self, val):
        """Slider moved → update threshold and re-run the check."""
        pct = int(float(val))
        self.thr_label.config(text=f'{pct} %')
        self.threshold = pct / 100.0
        self._run_check()

    def _run_check(self):
        """
        Run correct_morse_message on the current self._morse string and
        refresh the correction rows.
        """
        self.status_var.set('Checking Morse patterns…')
        self.cur_label.config(text=self._morse)
        result = correct_morse_message(self._morse, threshold=self.threshold)
        self._build_correction_rows(result)

    def _apply_suggestion(self, char: str, sugg_morse: str, word_idx: int, tok_idx: int):
        """
        User clicked a suggestion button.

        Replaces the token in self._morse, updates the parent input field
        immediately, and re-runs the correction check so the dialog refreshes.
        """
        log.info(
            "[Correction Dialog] User selected: %s (%s) for word=%d token=%d  input=%r",
            char, sugg_morse, word_idx, tok_idx, self._morse
        )
        # Apply the replacement
        self._morse = apply_correction(self._morse, word_idx, tok_idx, sugg_morse)

        # Update parent text field immediately so the user sees progress
        self.parent_page.txt_input.delete('1.0', 'end')
        self.parent_page.txt_input.insert('1.0', self._morse)

        # Re-run check to refresh the dialog
        self._run_check()

    def _apply_and_close(self):
        """
        Write the current (corrected) Morse back to the parent's input field
        and close the dialog.
        """
        self.parent_page.txt_input.delete('1.0', 'end')
        self.parent_page.txt_input.insert('1.0', self._morse)
        self.parent_page.status_var.set(
            'Corrections applied. Click DECODE to decode.'
        )
        log.info("[Correction Dialog] Closed. Final morse: %r", self._morse)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Decoder Page  (extends original with CHECK & CORRECT button)
# ─────────────────────────────────────────────────────────────────────────────

class DecoderPage(tk.Frame):
    """Morse to Text AI Decoder page.

    All original functionality is preserved.
    The only additions are:
      • A 'CHECK & CORRECT' button in the control row.
      • _check_and_correct() method that opens MorseCorrectionDialog.
    """

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._correction_threshold = _DEFAULT_THRESHOLD
        self._build()

    def _build(self):
        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=C['bg'])
        hdr.pack(fill='x', padx=28, pady=(24, 0))
        tk.Label(hdr, text='MORSE CODE  →  TEXT (AI)', bg=C['bg'],
                 fg=C['accent'], font=('Segoe UI', 11, 'bold')).pack(side='left')
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=28, pady=(10, 0))

        tk.Label(self,
                 text='Decode Morse code to text using the trained AI model.',
                 bg=C['bg'], fg=C['text2'], font=('Segoe UI', 10)
                 ).pack(anchor='w', padx=28, pady=(10, 0))

        # ── AI Model Status bar ───────────────────────────────────────────
        model_bar = tk.Frame(self, bg=C['card2'],
                             highlightbackground=C['border'], highlightthickness=1)
        model_bar.pack(fill='x', padx=28, pady=(14, 0))

        ml_inner = tk.Frame(model_bar, bg=C['card2'])
        ml_inner.pack(fill='x', padx=16, pady=10)

        tk.Label(ml_inner, text='AI MODEL', bg=C['card2'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        self.model_status_canvas = tk.Canvas(ml_inner, width=10, height=10,
                                             bg=C['card2'], bd=0, highlightthickness=0)
        self.model_status_canvas.pack(side='left', padx=(12, 6))
        self._led = self.model_status_canvas.create_oval(1, 1, 9, 9,
                                                         fill=C['error'], outline='')

        self.model_status_var = tk.StringVar(value='Not Loaded')
        tk.Label(ml_inner, textvariable=self.model_status_var,
                 bg=C['card2'], fg=C['error'], font=('Segoe UI', 9)
                 ).pack(side='left', padx=(0, 20))

        _btn(ml_inner, '  LOAD AI MODEL  ', self._load_model,
             C['purple']).pack(side='left')

        tk.Label(ml_inner, text='Model path: models/morse_decoder_v2.keras',
                 bg=C['card2'], fg=C['text3'], font=('Segoe UI', 8)).pack(side='right')

        # ── Main panels ───────────────────────────────────────────────────
        body = tk.Frame(self, bg=C['bg'])
        body.pack(fill='both', expand=True, padx=20, pady=14)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(1, weight=1)

        # Input
        tk.Label(body, text='MORSE CODE INPUT', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')
                 ).grid(row=0, column=0, sticky='w', padx=8, pady=(0, 4))

        in_frame = tk.Frame(body, bg=C['card'],
                            highlightbackground=C['border'], highlightthickness=1)
        in_frame.grid(row=1, column=0, sticky='nsew', padx=8)

        self.txt_input = tk.Text(in_frame, bg=C['card'], fg=C['success'],
                                 font=('Consolas', 13), bd=0, padx=14, pady=12,
                                 insertbackground=C['accent'], wrap='word',
                                 selectbackground=C['accent'],
                                 highlightthickness=0)
        sb_in = ttk.Scrollbar(in_frame, orient='vertical', command=self.txt_input.yview)
        self.txt_input.configure(yscrollcommand=sb_in.set)
        self.txt_input.pack(side='left', fill='both', expand=True)
        sb_in.pack(side='right', fill='y')

        # Output
        tk.Label(body, text='DECODED TEXT OUTPUT', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')
                 ).grid(row=0, column=1, sticky='w', padx=8, pady=(0, 4))

        out_frame = tk.Frame(body, bg=C['input_bg'],
                             highlightbackground=C['border'], highlightthickness=1)
        out_frame.grid(row=1, column=1, sticky='nsew', padx=8)

        self.txt_output = tk.Text(out_frame, bg=C['input_bg'], fg=C['text1'],
                                  font=('Segoe UI', 12), bd=0, padx=14, pady=12,
                                  state='disabled', wrap='word',
                                  selectbackground=C['accent'],
                                  highlightthickness=0)
        sb_out = ttk.Scrollbar(out_frame, orient='vertical', command=self.txt_output.yview)
        self.txt_output.configure(yscrollcommand=sb_out.set)
        self.txt_output.pack(side='left', fill='both', expand=True)
        sb_out.pack(side='right', fill='y')

        # ── Controls ──────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=C['bg'])
        ctrl.pack(fill='x', padx=28, pady=(0, 12))

        # Original buttons (unchanged)
        _btn(ctrl, '   DECODE   ', self._decode, C['accent']).pack(side='left', padx=(0, 8))

        # ── NEW: Check & Correct button ───────────────────────────────────
        _btn(ctrl, '  ⚡ CHECK & CORRECT  ', self._check_and_correct,
             C['warning'], C['bg']).pack(side='left', padx=(0, 8))

        _btn(ctrl, '   COPY   ', self._copy, '#21262D', C['text1']).pack(side='left', padx=(0, 8))
        _btn(ctrl, '   CLEAR   ', self._clear, C['error']).pack(side='left')

        # ── Confidence bar ────────────────────────────────────────────────
        conf_frame = tk.Frame(self, bg=C['card2'],
                              highlightbackground=C['border'], highlightthickness=1)
        conf_frame.pack(fill='x', padx=28, pady=(0, 20))

        ci = tk.Frame(conf_frame, bg=C['card2'])
        ci.pack(fill='x', padx=16, pady=10)

        tk.Label(ci, text='AI CONFIDENCE', bg=C['card2'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')

        self.conf_var = tk.StringVar(value='N/A')
        self.conf_label = tk.Label(ci, textvariable=self.conf_var, bg=C['card2'],
                                   fg=C['text3'], font=('Consolas', 13, 'bold'))
        self.conf_label.pack(side='left', padx=16)

        self.pred_var = tk.StringVar(value='No prediction yet.')
        self.pred_label = tk.Label(ci, textvariable=self.pred_var, bg=C['card2'],
                                   fg=C['text3'], font=('Segoe UI', 9))
        self.pred_label.pack(side='left')

        self.status_var = tk.StringVar(value='Ready.')
        self.status_label = tk.Label(conf_frame, textvariable=self.status_var,
                                     bg=C['card2'], fg=C['text3'], font=('Segoe UI', 9))
        self.status_label.pack(anchor='e', padx=16, pady=(0, 6))

    # ─── Original actions (completely unchanged) ──────────────────────────

    def _load_model(self):
        """Load the AI model in a background thread and update UI with the real result."""
        self.model_status_var.set('Loading...')
        self.model_status_canvas.itemconfig(self._led, fill=C['warning'])
        self.status_var.set('Loading AI model – please wait...')

        def _do():
            ok, msg = load_ai_model()

            def _update():
                if ok:
                    self.model_status_var.set('AI Loaded')
                    self.model_status_canvas.itemconfig(self._led, fill=C['success'])
                    for w in self.model_status_canvas.master.winfo_children():
                        if isinstance(w, tk.Label) and w.cget('textvariable') != '':
                            try:
                                if str(w.cget('textvariable')) == str(self.model_status_var):
                                    w.config(fg=C['success'])
                            except Exception:
                                pass
                    self.status_var.set('AI model loaded successfully.')
                else:
                    first_line = msg.split('\n')[0] if msg else 'Unknown error'
                    short = first_line[:60] + ('...' if len(first_line) > 60 else '')
                    self.model_status_var.set(f'Failed: {short}')
                    self.model_status_canvas.itemconfig(self._led, fill=C['error'])
                    for w in self.model_status_canvas.master.winfo_children():
                        if isinstance(w, tk.Label):
                            try:
                                if str(w.cget('textvariable')) == str(self.model_status_var):
                                    w.config(fg=C['error'])
                            except Exception:
                                pass
                    self.status_var.set(f'Load failed: {first_line}')
                    messagebox.showerror(
                        'Failed to Load AI Model',
                        f'Failed to load model:\n\n{msg}',
                        parent=self,
                    )

            self.after(0, _update)

        threading.Thread(target=_do, daemon=True).start()

    def _decode(self):
        morse = self.txt_input.get('1.0', 'end').strip()
        if not morse:
            self.status_var.set('Error: Morse input is empty.')
            return

        result = decode_morse(morse)
        text, confidence, elapsed, method = result

        self.txt_output.configure(state='normal')
        self.txt_output.delete('1.0', 'end')
        self.txt_output.insert('1.0', text)
        self.txt_output.configure(state='disabled')

        elapsed_ms = elapsed * 1000

        if is_model_loaded():
            # AI model was used – show real confidence percentage
            pct = confidence * 100
            if text == 'Prediction failed':
                self.conf_var.set('Err')
                self.conf_label.config(fg=C['error'])
                self.pred_label.config(fg=C['error'])
                self.pred_var.set(f'Prediction failed  |  {method}  |  {elapsed_ms:.1f} ms')
                self.status_var.set('AI inference failed. Check console for details.')
            elif pct < 60.0:
                self.conf_var.set(f'{pct:.2f}%')
                self.conf_label.config(fg=C['warning'])
                self.pred_label.config(fg=C['warning'])
                self.pred_var.set(f'Low confidence  |  {method}  |  {elapsed_ms:.1f} ms')
                self.status_var.set(f'Decoded {len(morse)} chars — low confidence ({pct:.1f}%).')
            else:
                self.conf_var.set(f'{pct:.2f}%')
                self.conf_label.config(fg=C['success'])
                self.pred_label.config(fg=C['text3'])
                self.pred_var.set(f'AI prediction  |  {pct:.1f}% confidence  |  {elapsed_ms:.1f} ms')
                self.status_var.set(f'Decoded {len(morse)} characters successfully.')
        else:
            # AI model not loaded – dictionary fallback; confidence is meaningless
            self.conf_var.set('N/A')
            self.conf_label.config(fg=C['text3'])
            self.pred_label.config(fg=C['text3'])
            self.pred_var.set(
                f'Dictionary fallback  |  {elapsed_ms:.1f} ms  '
                f'(Load AI Model for real inference).'
            )
            self.status_var.set(f'Decoded {len(morse)} characters using dictionary.')

    def _copy(self):
        out = self.txt_output.get('1.0', 'end').strip()
        if out:
            self.clipboard_clear()
            self.clipboard_append(out)
            self.status_var.set('Decoded text copied to clipboard.')

    def _clear(self):
        self.txt_input.delete('1.0', 'end')
        self.txt_output.configure(state='normal')
        self.txt_output.delete('1.0', 'end')
        self.txt_output.configure(state='disabled')
        self.conf_var.set('N/A')
        self.conf_label.config(fg=C['text3'])
        self.pred_var.set('No prediction yet.')
        self.status_var.set('Cleared.')

    # ─── New: Check & Correct action ─────────────────────────────────────

    def _check_and_correct(self):
        """
        Open the MorseCorrectionDialog for the current input.

        Flow:
          1. Read the current Morse string from the input field.
          2. Run correct_morse_message() via MorseCorrectionDialog.
          3. The dialog lets the user select corrections interactively.
          4. On Apply, the corrected Morse is written back to txt_input.
          5. The user then clicks DECODE as normal.

        The AI model is not involved in this step.
        """
        morse = self.txt_input.get('1.0', 'end').strip()
        if not morse:
            self.status_var.set('Error: Morse input is empty.')
            return

        log.info("[DecoderPage] Opening correction dialog for: %r", morse[:80])
        self.status_var.set('Opening correction dialog…')

        # MorseCorrectionDialog is modal – it blocks until the user closes it
        MorseCorrectionDialog(self, morse, self._correction_threshold)
