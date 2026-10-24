"""
LoRaShield – About Page
"""

import tkinter as tk
from tkinter import ttk
import os

C = {
    'bg': '#0D1117', 'card': '#161B22', 'card2': '#1C2128',
    'border': '#30363D', 'accent': '#2563EB', 'accent_h': '#1D4ED8',
    'text1': '#E6EDF3', 'text2': '#8B949E', 'text3': '#656D76',
    'success': '#3FB950',
}

TEAM_MEMBERS = [
    'Rakshitha K',
    'Shravana',
    'Soumya',
    'Soundarya',
    'Prashanth M',
]

GUIDE = 'Asst. Prof. Nayana T'


class AboutPage(tk.Frame):
    """About page displaying project information and team details."""

    def __init__(self, parent, app_state: dict, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self.app_state = app_state
        self._build()

    def _build(self):
        # Centered container
        canvas = tk.Canvas(self, bg=C['bg'], bd=0, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(fill='both', expand=True)

        body = tk.Frame(canvas, bg=C['bg'])
        canvas.create_window((0, 0), window=body, anchor='nw')
        body.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

        def mw(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', mw)

        # ── Logo / Header ─────────────────────────────────────────────────────
        top = tk.Frame(body, bg=C['bg'])
        top.pack(pady=(40, 0))

        # Shield icon (canvas-drawn)
        shield = tk.Canvas(top, width=80, height=90, bg=C['bg'],
                           bd=0, highlightthickness=0)
        shield.pack()
        # Draw shield shape
        pts = [40, 5, 75, 20, 75, 55, 40, 85, 5, 55, 5, 20]
        shield.create_polygon(pts, fill=C['accent'], outline=C['accent_h'], width=2)
        # Draw inner highlight
        pts2 = [40, 18, 63, 30, 63, 52, 40, 72, 17, 52, 17, 30]
        shield.create_polygon(pts2, fill=C['accent_h'], outline='', )
        # Radio waves
        for r in [12, 20, 28]:
            shield.create_arc(40 - r, 40 - r, 40 + r, 40 + r,
                              start=30, extent=120, style='arc',
                              outline='white', width=2)

        tk.Label(top, text='LoRaShield', bg=C['bg'],
                 fg=C['text1'], font=('Segoe UI', 28, 'bold')).pack(pady=(12, 0))

        tk.Label(top,
                 text='AI Enhanced Secure Morse Code\nCommunication System using LoRa Technology',
                 bg=C['bg'], fg=C['text2'], font=('Segoe UI', 12),
                 justify='center').pack(pady=(4, 0))

        tk.Frame(body, bg=C['accent'], height=2).pack(fill='x', padx=100, pady=20)

        # ── Cards row ─────────────────────────────────────────────────────────
        cards = tk.Frame(body, bg=C['bg'])
        cards.pack(pady=(0, 0), padx=60)
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        cards.columnconfigure(2, weight=1)

        info_items = [
            ('[>]', 'Text to Morse', 'Standard ITU Morse\nEncoding'),
            ('[*]', 'AI Decoding', 'TensorFlow / Keras\nNeural Network'),
            ('[#]', 'AES-256', 'Military-grade\nEncryption'),
        ]

        for col, (icon, title, sub) in enumerate(info_items):
            card = tk.Frame(cards, bg=C['card'],
                            highlightbackground=C['border'], highlightthickness=1)
            card.grid(row=0, column=col, padx=8, pady=0, sticky='nsew')

            tk.Frame(card, bg=C['accent'], height=3).pack(fill='x')
            inner = tk.Frame(card, bg=C['card'], padx=20, pady=16)
            inner.pack()
            tk.Label(inner, text=icon, bg=C['card'],
                     fg=C['accent'], font=('Segoe UI', 20)).pack()
            tk.Label(inner, text=title, bg=C['card'],
                     fg=C['text1'], font=('Segoe UI', 11, 'bold')).pack(pady=(6, 0))
            tk.Label(inner, text=sub, bg=C['card'],
                     fg=C['text3'], font=('Segoe UI', 9), justify='center').pack(pady=(4, 0))

        # ── Team section ─────────────────────────────────────────────────────
        tk.Frame(body, bg=C['border'], height=1).pack(fill='x', padx=60, pady=28)

        team_hdr = tk.Label(body, text='PROJECT TEAM', bg=C['bg'],
                            fg=C['text3'], font=('Segoe UI', 10, 'bold'))
        team_hdr.pack()

        team_frame = tk.Frame(body, bg=C['bg'])
        team_frame.pack(pady=(16, 0))

        for i, member in enumerate(TEAM_MEMBERS):
            m_card = tk.Frame(team_frame, bg=C['card2'],
                              highlightbackground=C['border'], highlightthickness=1)
            m_card.pack(fill='x', padx=60, pady=4)

            inner = tk.Frame(m_card, bg=C['card2'])
            inner.pack(fill='x', padx=20, pady=10)

            # Avatar circle with initials
            initials = ''.join(w[0] for w in member.split()[:2])
            av = tk.Canvas(inner, width=36, height=36, bg=C['card2'],
                           bd=0, highlightthickness=0)
            av.pack(side='left', padx=(0, 14))
            av.create_oval(2, 2, 34, 34, fill=C['accent'], outline='')
            av.create_text(18, 18, text=initials,
                           fill=C['text1'], font=('Segoe UI', 11, 'bold'))

            tk.Label(inner, text=member, bg=C['card2'],
                     fg=C['text1'], font=('Segoe UI', 11)).pack(side='left')

            tk.Label(inner, text='Team Member', bg=C['card2'],
                     fg=C['text3'], font=('Segoe UI', 9)).pack(side='right', padx=8)

        # ── Guide ─────────────────────────────────────────────────────────────
        tk.Frame(body, bg=C['border'], height=1).pack(fill='x', padx=60, pady=20)

        guide_card = tk.Frame(body, bg=C['card'],
                              highlightbackground=C['accent'], highlightthickness=1)
        guide_card.pack(padx=60, fill='x')
        gi = tk.Frame(guide_card, bg=C['card'])
        gi.pack(fill='x', padx=24, pady=14)

        tk.Label(gi, text='PROJECT GUIDE', bg=C['card'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack(side='left')
        tk.Label(gi, text=GUIDE, bg=C['card'],
                 fg=C['text1'], font=('Segoe UI', 11, 'bold')).pack(side='left', padx=16)
        tk.Label(gi, text='Faculty Mentor', bg=C['card'],
                 fg=C['text3'], font=('Segoe UI', 9)).pack(side='right')

        # ── Tech stack ────────────────────────────────────────────────────────
        tk.Frame(body, bg=C['border'], height=1).pack(fill='x', padx=60, pady=20)

        tk.Label(body, text='TECHNOLOGY STACK', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9, 'bold')).pack()

        tech = [
            ('Python 3', 'Core Language'),
            ('ttkbootstrap', 'Modern UI Framework'),
            ('TensorFlow / Keras', 'AI / Deep Learning'),
            ('AES-256', 'Encryption (pycryptodome)'),
            ('pyserial', 'ESP32 Communication'),
            ('LoRa SX1276', 'Radio Transceiver'),
        ]

        tech_frame = tk.Frame(body, bg=C['bg'])
        tech_frame.pack(padx=60, pady=(12, 0))

        for i, (name, role) in enumerate(tech):
            col = i % 3
            row = i // 3
            tech_frame.columnconfigure(col, weight=1)
            tc = tk.Frame(tech_frame, bg=C['card2'],
                          highlightbackground=C['border'], highlightthickness=1)
            tc.grid(row=row, column=col, padx=6, pady=4, sticky='ew')
            tk.Frame(tc, bg=C['accent'], width=3).pack(side='left', fill='y')
            ti = tk.Frame(tc, bg=C['card2'])
            ti.pack(side='left', padx=12, pady=8)
            tk.Label(ti, text=name, bg=C['card2'],
                     fg=C['text1'], font=('Segoe UI', 9, 'bold')).pack(anchor='w')
            tk.Label(ti, text=role, bg=C['card2'],
                     fg=C['text3'], font=('Segoe UI', 8)).pack(anchor='w')

        # ── Footer ────────────────────────────────────────────────────────────
        footer = tk.Frame(body, bg=C['bg'])
        footer.pack(pady=(30, 20))
        tk.Label(footer, text='© 2026 LoRaShield', bg=C['bg'],
                 fg=C['text3'], font=('Segoe UI', 9)).pack()
        tk.Label(footer,
                 text='Final Year Project  |  Department of Electronics & Communication Engineering',
                 bg=C['bg'], fg=C['text3'], font=('Segoe UI', 8)).pack()
