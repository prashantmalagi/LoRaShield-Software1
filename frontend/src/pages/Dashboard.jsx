import { useState, useEffect } from 'react'

const API = 'http://127.0.0.1:8000'

function ts() { return new Date().toLocaleTimeString('en-GB') }

export default function Dashboard() {
  const [data, setData] = useState(null)
  const [log, setLog] = useState([
    { t: ts(), msg: 'LoRaShield system initialized.' },
    { t: ts(), msg: 'Waiting for device connection…' },
  ])

  const addLog = (msg) => setLog(prev => [...prev.slice(-100), { t: ts(), msg }])

  const refresh = async () => {
    try {
      const r = await fetch(`${API}/api/dashboard`)
      setData(await r.json())
    } catch {}
  }

  useEffect(() => {
    refresh()
    const iv = setInterval(refresh, 2000)
    return () => clearInterval(iv)
  }, [])

  const STATUS = data ? [
    { label: 'Connection',  color: data.connected ? 'var(--success)' : 'var(--warning)', val: data.connected ? 'Connected' : 'Disconnected' },
    { label: 'AI Model',    color: data.ai_loaded ? 'var(--success)' : 'var(--error)',   val: data.ai_loaded ? 'AI Loaded' : 'Not Loaded' },
    { label: 'ESP32',       color: data.connected ? 'var(--success)' : 'var(--error)',   val: data.connected ? 'Online' : 'Offline' },
    { label: 'LoRa Module', color: data.connected ? 'var(--success)' : 'var(--error)',   val: data.connected ? 'Active' : 'Offline' },
    { label: 'Encryption',  color: data.crypto_available ? 'var(--success)' : 'var(--error)', val: data.crypto_available ? 'AES-256 Ready' : 'Unavailable' },
    { label: 'COM Port',    color: 'var(--text2)',  val: data.port || 'None' },
    { label: 'Signal',      color: 'var(--warning)', val: 'N/A' },
    { label: 'Battery',     color: 'var(--text3)',  val: 'N/A' },
  ] : []

  const METRICS = data ? [
    { title: 'Messages Sent',     value: data.msgs_sent,     sub: `Last: ${data.last_comm_time}`, color: 'var(--accent)',   icon: '→' },
    { title: 'Messages Received', value: data.msgs_received,  sub: `Records: ${data.record_count}`, color: 'var(--success)', icon: '←' },
    { title: 'AI Model',          value: data.ai_loaded ? 'ON' : 'N/A', sub: data.ai_loaded ? 'Model ready' : 'Not loaded', color: 'var(--purple)', icon: '★' },
    { title: 'Device Status',     value: (data.connected && data.crypto_available) ? 'READY' : 'IDLE',
      sub: (data.connected && data.crypto_available) ? 'Connected & encrypted' : 'Awaiting connection',
      color: 'var(--teal)', icon: '#' },
  ] : []

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-title">System Dashboard</span>
        <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
          {new Date().toLocaleString()}
        </span>
      </div>
      <div className="page-divider" />

      {/* Status grid */}
      <p className="section-label" style={{ marginTop: 0 }}>System Status</p>
      <div className="status-grid">
        {STATUS.map((s, i) => (
          <div className="status-cell" key={i}>
            <div className="status-cell-left">
              <div className="led" style={{ background: s.color, boxShadow: `0 0 6px ${s.color}` }} />
              {s.label}
            </div>
            <span className="status-cell-right" style={{ color: s.color }}>{s.val}</span>
          </div>
        ))}
      </div>

      {/* Metric cards */}
      <p className="section-label">Key Metrics</p>
      <div className="metric-grid">
        {METRICS.map((m, i) => (
          <div className="metric-card" key={i}>
            <div className="metric-stripe" style={{ background: m.color }} />
            <div className="metric-body">
              <div className="metric-icon" style={{ color: m.color }}>{m.icon}</div>
              <div className="metric-value">{m.value}</div>
              <div className="metric-title">{m.title}</div>
              <div className="metric-sub">{m.sub}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Activity log */}
      <p className="section-label">Activity Log</p>
      <div className="log-box">
        {log.map((l, i) => (
          <div className="log-line" key={i}>
            <span className="ts">[{l.t}] </span>{l.msg}
          </div>
        ))}
      </div>
    </div>
  )
}
