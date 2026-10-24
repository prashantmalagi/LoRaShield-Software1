"""
LoRaShield – Main Application Entry Point
==========================================
AI Enhanced Secure Morse Code Communication System using LoRa Technology

Run:
    python app.py

Requirements:
    pip install -r requirements.txt
"""

import tkinter as tk
from tkinter import ttk
import sys
import os
from pathlib import Path

# ── Ensure project root is on path ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Initialise logging before any other local imports ─────────────────────────
from utils.logger import get_logger   # noqa: E402
log = get_logger(__name__)
log.info("LoRaShield starting up.")

# ── Try ttkbootstrap; fall back to plain tkinter ──────────────────────────────
try:
    import ttkbootstrap as ttk_boot
    TTKBOOTSTRAP = True
except ImportError:
    TTKBOOTSTRAP = False

# ── Import pages ──────────────────────────────────────────────────────────────
from ui.dashboard   import DashboardPage
from ui.encoder     import EncoderPage
from ui.decoder     import DecoderPage
from ui.encryption  import EncryptionPage
from ui.lora        import LoRaPage
from ui.history     import HistoryPage
from ui.settings    import SettingsPage
from ui.about       import AboutPage

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    'bg':          '#0D1117',
    'sidebar':     '#010409',
    'sidebar_btn': '#0D1117',
    'sidebar_act': '#1A2332',
    'card':        '#161B22',
    'border':      '#21262D',
    'accent':      '#2563EB',
    'accent_h':    '#1D4ED8',
    'text1':       '#E6EDF3',
    'text2':       '#8B949E',
    'text3':       '#656D76',
    'success':     '#3FB950',
}

# ── Sidebar navigation items ──────────────────────────────────────────────────
NAV_ITEMS = [
    ('dashboard',   '[=]',  'Dashboard'),
    ('encoder',     '[>]',  'Text to Morse'),
    ('decoder',     '[<]',  'Morse to Text (AI)'),
    ('encryption',  '[#]',  'Encryption'),
    ('lora',        '[~]',  'LoRa Communication'),
    ('history',     '[-]',  'Message History'),
    ('settings',    '[o]',  'Settings'),
    ('about',       '[i]',  'About'),
]

APP_VERSION = 'v1.0.0'


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
class Sidebar(tk.Frame):
    """Left navigation sidebar with animated active-state indicator."""

    def __init__(self, parent, on_navigate, **kw):
        super().__init__(parent, bg=C['sidebar'], width=220, **kw)
        self.pack_propagate(False)
        self.on_navigate = on_navigate
        self._active_page: str = ''
        self._btn_map: dict[str, tk.Frame] = {}
        self._build()

    def _build(self):
        # ── Logo area ─────────────────────────────────────────────────────────
        logo_frame = tk.Frame(self, bg=C['sidebar'])
        logo_frame.pack(fill='x', pady=(0, 0))

        # Shield canvas logo
        shield_cv = tk.Canvas(logo_frame, width=44, height=48,
                               bg=C['sidebar'], bd=0, highlightthickness=0)
        shield_cv.pack(side='left', padx=(18, 10), pady=18)
        pts = [22, 4, 40, 14, 40, 32, 22, 44, 4, 32, 4, 14]
        shield_cv.create_polygon(pts, fill=C['accent'], outline='#3B82F6', width=1)
        pts2 = [22, 13, 32, 20, 32, 31, 22, 38, 12, 31, 12, 20]
        shield_cv.create_polygon(pts2, fill=C['accent_h'], outline='')
        for r in [6, 11, 16]:
            shield_cv.create_arc(22 - r, 24 - r, 22 + r, 24 + r,
                                  start=30, extent=120, style='arc',
                                  outline='white', width=1)

        title_f = tk.Frame(logo_frame, bg=C['sidebar'])
        title_f.pack(side='left', pady=18)
        tk.Label(title_f, text='LoRaShield', bg=C['sidebar'],
                 fg=C['text1'], font=('Segoe UI', 14, 'bold')).pack(anchor='w')
        tk.Label(title_f, text=APP_VERSION, bg=C['sidebar'],
                 fg=C['text3'], font=('Segoe UI', 8)).pack(anchor='w')

        # Separator
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x', padx=0)

        # ── Navigation buttons ────────────────────────────────────────────────
        nav_scroll = tk.Frame(self, bg=C['sidebar'])
        nav_scroll.pack(fill='both', expand=True, pady=8)

        for page_key, icon, label in NAV_ITEMS:
            self._make_nav_btn(nav_scroll, page_key, icon, label)

        # ── Bottom status area ────────────────────────────────────────────────
        tk.Frame(self, bg=C['border'], height=1).pack(fill='x')

        bottom = tk.Frame(self, bg=C['sidebar'], pady=10)
        bottom.pack(fill='x')

        status_row = tk.Frame(bottom, bg=C['sidebar'])
        status_row.pack(fill='x', padx=16)

        # Connection LED
        self.conn_cv = tk.Canvas(status_row, width=10, height=10,
                                  bg=C['sidebar'], bd=0, highlightthickness=0)
        self.conn_cv.pack(side='left')
        self._conn_led = self.conn_cv.create_oval(1, 1, 9, 9,
                                                   fill=C['text3'], outline='')
        self.conn_lbl = tk.Label(status_row, text='No Device', bg=C['sidebar'],
                                  fg=C['text3'], font=('Segoe UI', 8))
        self.conn_lbl.pack(side='left', padx=(6, 0))

        tk.Label(bottom, text='© 2026 LoRaShield', bg=C['sidebar'],
                 fg=C['text3'], font=('Segoe UI', 7)).pack(pady=(6, 0))

    def _make_nav_btn(self, parent: tk.Frame, page_key: str, icon: str, label: str):
        """Create one sidebar navigation button with hover + active effects."""
        container = tk.Frame(parent, bg=C['sidebar'], cursor='hand2')
        container.pack(fill='x')

        # Active indicator stripe (left edge)
        stripe = tk.Frame(container, width=3, bg=C['sidebar'])
        stripe.pack(side='left', fill='y', ipady=4)

        inner = tk.Frame(container, bg=C['sidebar'])
        inner.pack(side='left', fill='x', expand=True, padx=(8, 16), pady=2)

        icon_lbl = tk.Label(inner, text=icon, bg=C['sidebar'],
                            fg=C['text3'], font=('Consolas', 13))
        icon_lbl.pack(side='left', padx=(0, 10))

        text_lbl = tk.Label(inner, text=label, bg=C['sidebar'],
                            fg=C['text3'], font=('Segoe UI', 10))
        text_lbl.pack(side='left', anchor='w', pady=8)

        def on_click(pk=page_key):
            self.on_navigate(pk)

        def on_enter(event, c=container, s=stripe, il=icon_lbl, tl=text_lbl, pk=page_key):
            if pk != self._active_page:
                c.config(bg=C['sidebar_btn'])
                s.config(bg=C['sidebar_btn'])
                il.config(bg=C['sidebar_btn'])
                tl.config(bg=C['sidebar_btn'])

        def on_leave(event, c=container, s=stripe, il=icon_lbl, tl=text_lbl, pk=page_key):
            if pk != self._active_page:
                c.config(bg=C['sidebar'])
                s.config(bg=C['sidebar'])
                il.config(bg=C['sidebar'])
                tl.config(bg=C['sidebar'])

        for widget in (container, inner, icon_lbl, text_lbl):
            widget.bind('<Button-1>', lambda e, pk=page_key: on_click(pk))
            widget.bind('<Enter>', on_enter)
            widget.bind('<Leave>', on_leave)

        self._btn_map[page_key] = {
            'container': container, 'stripe': stripe,
            'icon': icon_lbl, 'text': text_lbl,
        }

    def set_active(self, page_key: str):
        """Highlight the active navigation button."""
        # Deactivate old
        if self._active_page and self._active_page in self._btn_map:
            old = self._btn_map[self._active_page]
            for w in (old['container'], old['stripe'], old['icon'], old['text']):
                w.config(bg=C['sidebar'])
            old['stripe'].config(bg=C['sidebar'])
            old['text'].config(fg=C['text3'])
            old['icon'].config(fg=C['text3'])

        # Activate new
        self._active_page = page_key
        if page_key in self._btn_map:
            new = self._btn_map[page_key]
            for w in (new['container'], new['icon'], new['text']):
                w.config(bg=C['sidebar_act'])
            new['stripe'].config(bg=C['accent'])
            new['text'].config(fg=C['text1'])
            new['icon'].config(fg=C['accent'])

    def set_connection_status(self, connected: bool, port: str = ''):
        color = C['success'] if connected else C['text3']
        text = port if connected else 'No Device'
        self.conn_cv.itemconfig(self._conn_led, fill=color)
        self.conn_lbl.config(text=text, fg=color)


# ══════════════════════════════════════════════════════════════════════════════
# Content area (page switcher with fade animation)
# ══════════════════════════════════════════════════════════════════════════════
class ContentArea(tk.Frame):
    """Hosts pages and handles animated page transitions."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C['bg'], **kw)
        self._pages: dict[str, tk.Frame] = {}
        self._current: tk.Frame | None = None

    def register_page(self, key: str, page: tk.Frame):
        """Register a page widget under a key."""
        page.place(x=0, y=0, relwidth=1, relheight=1)
        page.lower()
        self._pages[key] = page

    def show_page(self, key: str):
        """Bring the named page to the front with a simple reveal."""
        if key not in self._pages:
            return
        page = self._pages[key]
        if self._current is page:
            return
        if self._current:
            self._current.lower()
        page.lift()
        self._current = page


# ══════════════════════════════════════════════════════════════════════════════
# Main Application
# ══════════════════════════════════════════════════════════════════════════════
class LoRaShieldApp:
    """Main application controller."""

    def __init__(self):
        # ── Window setup ──────────────────────────────────────────────────────
        if TTKBOOTSTRAP:
            self.root = ttk_boot.Window(themename='darkly')
        else:
            self.root = tk.Tk()

        self.root.title('LoRaShield')
        self.root.geometry('1280x800')
        self.root.minsize(1000, 650)
        self.root.configure(bg=C['bg'])

        try:
            self.root.iconbitmap('')
        except Exception:
            pass

        # ── Global state ──────────────────────────────────────────────────────
        self.app_state: dict = {
            'com_port':    '',
            'enc_key':     'LoRaShield2026',
            'ai_loaded':   False,
            'connected':   False,
            'baud_rate':   9600,
            'auto_connect': False,
        }

        # ── Apply custom styles ───────────────────────────────────────────────
        self._apply_styles()

        # ── Layout ────────────────────────────────────────────────────────────
        self._build_layout()

        # ── Load settings ─────────────────────────────────────────────────────
        self._load_initial_settings()

        # ── Auto-connect if configured ────────────────────────────────────────
        self._auto_connect()

        # ── Sidebar LED polling ────────────────────────────────────────────────
        self._poll_sidebar_led()

        # ── Navigate to dashboard ─────────────────────────────────────────────
        self._navigate('dashboard')

    def _apply_styles(self):
        """Apply custom ttk styles for a consistent dark theme."""
        style = ttk.Style(self.root)
        try:
            style.theme_use('clam')
        except Exception:
            pass

        style.configure('TScrollbar',
                         background=C['card'], troughcolor=C['bg'],
                         borderwidth=0, arrowcolor=C['text3'])
        style.configure('TCombobox',
                         fieldbackground=C['card'], background=C['card'],
                         foreground=C['text1'], arrowcolor=C['text2'],
                         borderwidth=1, relief='flat')
        style.map('TCombobox',
                   fieldbackground=[('readonly', C['card'])],
                   foreground=[('readonly', C['text1'])],
                   selectbackground=[('readonly', C['accent'])])
        style.configure('TEntry',
                         fieldbackground=C['card'], foreground=C['text1'],
                         borderwidth=1, relief='flat')

    def _build_layout(self):
        """Construct the sidebar + content area layout."""
        # Main horizontal split
        main = tk.Frame(self.root, bg=C['bg'])
        main.pack(fill='both', expand=True)

        # Sidebar
        self.sidebar = Sidebar(main, on_navigate=self._navigate)
        self.sidebar.pack(side='left', fill='y')

        # Vertical separator
        tk.Frame(main, bg=C['border'], width=1).pack(side='left', fill='y')

        # Content area
        self.content = ContentArea(main)
        self.content.pack(side='left', fill='both', expand=True)

        # ── Register all pages ────────────────────────────────────────────────
        pages = {
            'dashboard':  DashboardPage,
            'encoder':    EncoderPage,
            'decoder':    DecoderPage,
            'encryption': EncryptionPage,
            'lora':       LoRaPage,
            'history':    HistoryPage,
            'settings':   SettingsPage,
            'about':      AboutPage,
        }

        for key, PageClass in pages.items():
            page = PageClass(self.content, self.app_state)
            self.content.register_page(key, page)

        self.dashboard_page: DashboardPage = self.content._pages['dashboard']

    def _navigate(self, page_key: str):
        """Switch to a page and update sidebar highlight."""
        self.content.show_page(page_key)
        self.sidebar.set_active(page_key)

        # Refresh history when visiting that page
        if page_key == 'history':
            history_page: HistoryPage = self.content._pages['history']
            history_page.refresh()

        self.dashboard_page.add_log(f'Navigated to: {page_key.replace("_", " ").title()}')

    def _load_initial_settings(self):
        """Load persisted settings into app_state on startup."""
        import json
        settings_file = ROOT / 'data' / 'settings.json'
        if settings_file.exists():
            try:
                with settings_file.open() as fh:
                    data = json.load(fh)
                self.app_state['com_port']    = data.get('com_port', '')
                self.app_state['enc_key']     = data.get('enc_key', 'LoRaShield2026')
                self.app_state['baud_rate']   = int(data.get('baud_rate', 9600))
                self.app_state['auto_connect'] = bool(data.get('auto_connect', False))
                log.info("Settings loaded from %s.", settings_file)
            except Exception as exc:
                log.warning("Could not load settings: %s", exc)

    def _auto_connect(self):
        """If auto_connect is enabled in settings, open the saved COM port."""
        from utils import serial_comm
        if not self.app_state.get('auto_connect'):
            return
        port = self.app_state.get('com_port', '')
        baud = self.app_state.get('baud_rate', 9600)
        if not port:
            return
        log.info("Auto-connecting to %s at %d baud.", port, baud)
        ok, msg = serial_comm.connect_esp32(port, baud)
        if ok:
            log.info("Auto-connect succeeded: %s", msg)
            self.dashboard_page.add_log(f'Auto-connected: {msg}')
        else:
            log.warning("Auto-connect failed: %s", msg)
            self.dashboard_page.add_log(f'Auto-connect failed: {msg}')

    def _poll_sidebar_led(self):
        """Update the sidebar connection LED every 2 s to reflect serial state."""
        from utils import serial_comm
        connected = serial_comm.is_connected()
        port = self.app_state.get('com_port', '')
        self.sidebar.set_connection_status(connected, port)
        self.root.after(2000, self._poll_sidebar_led)

    def run(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app = LoRaShieldApp()
    app.run()
