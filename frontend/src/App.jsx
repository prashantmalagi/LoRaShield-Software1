import { useState, useCallback } from 'react'
import './index.css'
import Dashboard from './pages/Dashboard'
import Encoder from './pages/Encoder'
import Decoder from './pages/Decoder'
import Encryption from './pages/Encryption'
import LoRa from './pages/LoRa'
import History from './pages/History'
import Settings from './pages/Settings'
import About from './pages/About'
import { useSerial } from './hooks/useSerial'

const NAV = [
  { key: 'dashboard',  icon: '⊞',  label: 'Dashboard'        },
  { key: 'encoder',    icon: '→',  label: 'Text to Morse'    },
  { key: 'decoder',    icon: '←',  label: 'Morse to Text'    },
  { key: 'encryption', icon: '#',  label: 'Encryption'       },
  { key: 'lora',       icon: '~',  label: 'LoRa Terminal'    },
  { key: 'history',    icon: '☰',  label: 'Message History'  },
  { key: 'settings',   icon: '⚙',  label: 'Settings'         },
  { key: 'about',      icon: 'ℹ',  label: 'About'            },
]

const PAGES = { Dashboard, Encoder, Decoder, Encryption, LoRa: LoRa, History, Settings, About }

function ShieldLogo() {
  return (
    <svg className="logo-shield" viewBox="0 0 44 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <polygon points="22,4 40,14 40,32 22,44 4,32 4,14" fill="#2563EB" stroke="#3B82F6" strokeWidth="1"/>
      <polygon points="22,13 32,20 32,31 22,38 12,31 12,20" fill="#1D4ED8"/>
      <path d="M22 18 A6 6 0 0 1 28 24" stroke="white" strokeWidth="1" fill="none"/>
      <path d="M22 15 A9 9 0 0 1 31 24" stroke="white" strokeWidth="1" fill="none"/>
      <path d="M22 12 A12 12 0 0 1 34 24" stroke="white" strokeWidth="1" fill="none"/>
    </svg>
  )
}

export default function App() {
  const [page, setPage] = useState('dashboard')
  const { connected, port } = useSerial()

  const PageComponent = PAGES[page.charAt(0).toUpperCase() + page.slice(1)] || Dashboard

  return (
    <div className="app-shell">
      {/* ── Sidebar ──────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <ShieldLogo />
          <div className="logo-text">
            <h1>LoRaShield</h1>
            <span>v1.0.0</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          {NAV.map(({ key, icon, label }) => (
            <div
              key={key}
              className={`nav-item ${page === key ? 'active' : ''}`}
              onClick={() => setPage(key)}
            >
              <span className="nav-icon">{icon}</span>
              {label}
            </div>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="conn-status-row">
            <div className={`led ${connected ? 'connected' : 'disconnected'}`} />
            <span>{connected ? port || 'Connected' : 'No Device'}</span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text3)', marginTop: 6 }}>
            © 2026 LoRaShield
          </div>
        </div>
      </aside>

      {/* ── Content ──────────────────────────────────────────────────── */}
      <main className="content-area">
        <div className="fade-in" key={page}>
          <PageComponent />
        </div>
      </main>
    </div>
  )
}
