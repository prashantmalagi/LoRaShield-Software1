import { useState, useEffect } from 'react'

const API = 'http://127.0.0.1:8000'

function Field({ label, children }) {
  return (
    <div className="field">
      <p className="input-label">{label}</p>
      {children}
    </div>
  )
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <p className="section-label" style={{ marginTop: 0 }}>{title}</p>
      <div className="card2" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        {children}
      </div>
    </div>
  )
}

export default function Settings() {
  const [s, setS] = useState({
    theme: 'Dark',
    com_port: '',
    baud_rate: '9600',
    auto_connect: false,
    ai_threshold: 0.75,
    enc_key: '',
  })
  const [showKey, setShowKey] = useState(false)
  const [status, setStatus] = useState('')

  useEffect(() => {
    fetch(`${API}/api/settings`).then(r => r.json()).then(d => {
      setS(prev => ({ ...prev, ...d }))
    }).catch(() => {})
  }, [])

  const update = (k, v) => setS(prev => ({ ...prev, [k]: v }))

  const save = async () => {
    try {
      const r = await fetch(`${API}/api/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(s),
      })
      const d = await r.json()
      setStatus(d.ok ? 'Settings saved successfully.' : `Error: ${d.message}`)
    } catch (e) { setStatus(`Error: ${e.message}`) }
  }

  const reset = () => {
    setS({ theme: 'Dark', com_port: '', baud_rate: '9600', auto_connect: false, ai_threshold: 0.75, enc_key: '' })
    setStatus('Defaults restored.')
  }

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-title">Settings</span>
      </div>
      <div className="page-divider" />

      <Section title="Interface">
        <Field label="Theme">
          <select className="input" style={{ width: 180 }} value={s.theme} onChange={e => update('theme', e.target.value)}>
            {['Dark', 'Dark Blue', 'Dark Slate'].map(t => <option key={t}>{t}</option>)}
          </select>
        </Field>
      </Section>

      <Section title="Hardware">
        <Field label="Default COM Port">
          <input className="input" style={{ width: 140 }} value={s.com_port} onChange={e => update('com_port', e.target.value)} placeholder="e.g. COM3" />
        </Field>
        <Field label="Baud Rate">
          <select className="input" style={{ width: 140 }} value={s.baud_rate} onChange={e => update('baud_rate', e.target.value)}>
            {['9600','19200','38400','57600','115200'].map(b => <option key={b}>{b}</option>)}
          </select>
        </Field>
        <Field label="Auto Connect on Start">
          <label className="flex items-center gap-2" style={{ cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={s.auto_connect}
              onChange={e => update('auto_connect', e.target.checked)}
              style={{ accentColor: 'var(--accent)', width: 16, height: 16 }}
            />
            <span style={{ fontSize: 12, color: 'var(--text2)' }}>Enable auto-connect on startup</span>
          </label>
        </Field>
      </Section>

      <Section title="AI Configuration">
        <Field label={`Confidence Threshold — ${s.ai_threshold.toFixed(2)}`}>
          <input
            type="range"
            min="0" max="1" step="0.01"
            value={s.ai_threshold}
            onChange={e => update('ai_threshold', parseFloat(e.target.value))}
            style={{ accentColor: 'var(--accent)', width: 240 }}
          />
        </Field>
      </Section>

      <Section title="Security">
        <Field label="Default Encryption Key">
          <div className="input-row">
            <input
              className="input input-mono"
              type={showKey ? 'text' : 'password'}
              value={s.enc_key}
              onChange={e => update('enc_key', e.target.value)}
              placeholder="Enter encryption key…"
            />
            <button className="btn btn-ghost btn-sm" onClick={() => setShowKey(v => !v)}>
              {showKey ? 'Hide' : 'Show'}
            </button>
          </div>
        </Field>
      </Section>

      {/* Save / Reset */}
      <div className="flex gap-2 items-center" style={{ marginTop: 8 }}>
        <button className="btn btn-primary" onClick={save}>Save Settings</button>
        <button className="btn btn-ghost" onClick={reset}>Reset Defaults</button>
        {status && <span style={{ fontSize: 12, color: status.startsWith('Error') ? 'var(--error)' : 'var(--success)', marginLeft: 8 }}>{status}</span>}
      </div>
    </div>
  )
}
