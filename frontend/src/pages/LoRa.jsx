import { useState, useEffect, useRef } from 'react'
import { useSerialMessages } from '../hooks/useSerial'

const API = 'http://127.0.0.1:8000'
function ts() { return new Date().toLocaleTimeString('en-GB') }

export default function LoRa() {
  const [ports, setPorts] = useState([])
  const [port, setPort] = useState('')
  const [baud, setBaud] = useState('9600')
  const [connected, setConnected] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [sendText, setSendText] = useState('')
  const [connLog, setConnLog] = useState([])
  const messages = useSerialMessages()
  const termRef = useRef(null)

  const addConnLog = (msg) => setConnLog(prev => [...prev.slice(-80), { t: ts(), msg }])

  const refreshPorts = async () => {
    try {
      const r = await fetch(`${API}/api/serial/ports`)
      const d = await r.json()
      setPorts(d.ports || [])
      if (d.ports?.length && !port) setPort(d.ports[0])
    } catch {}
  }

  const checkStatus = async () => {
    try {
      const r = await fetch(`${API}/api/serial/status`)
      const d = await r.json()
      setConnected(d.connected)
    } catch {}
  }

  useEffect(() => {
    refreshPorts()
    checkStatus()
    const iv = setInterval(checkStatus, 3000)
    return () => clearInterval(iv)
  }, [])

  // Auto-scroll terminal
  useEffect(() => {
    if (termRef.current) termRef.current.scrollTop = termRef.current.scrollHeight
  }, [messages])

  const connect = async () => {
    if (!port) { addConnLog('Error: No port selected.'); return }
    setConnecting(true)
    try {
      const r = await fetch(`${API}/api/serial/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ port, baud_rate: parseInt(baud) }),
      })
      const d = await r.json()
      addConnLog(d.message)
      setConnected(d.ok)
    } catch (e) { addConnLog(`Error: ${e.message}`) }
    setConnecting(false)
  }

  const disconnect = async () => {
    try {
      const r = await fetch(`${API}/api/serial/disconnect`, { method: 'POST' })
      const d = await r.json()
      addConnLog(d.message)
      setConnected(false)
    } catch (e) { addConnLog(`Error: ${e.message}`) }
  }

  const send = async () => {
    const msg = sendText.trim()
    if (!msg) return
    try {
      const r = await fetch(`${API}/api/serial/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, use_encryption: true }),
      })
      const d = await r.json()
      if (!d.ok) addConnLog(`Send error: ${d.message}`)
      else setSendText('')
    } catch (e) { addConnLog(`Error: ${e.message}`) }
  }

  const handleKeyDown = (e) => { if (e.key === 'Enter') send() }

  return (
    <div className="page" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div className="page-header">
        <span className="page-title">LoRa Communication</span>
        <div className="flex items-center gap-2">
          <div className={`led ${connected ? 'connected led-pulse' : 'disconnected'}`} />
          <span style={{ fontSize: 12, color: connected ? 'var(--success)' : 'var(--error)', fontWeight: 600 }}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
      <div className="page-divider" />

      {/* Config bar */}
      <div className="config-bar">
        <div className="config-group">
          <span className="config-label">COM Port</span>
          <select className="input" style={{ width: 120 }} value={port} onChange={e => setPort(e.target.value)}>
            {ports.map(p => <option key={p}>{p}</option>)}
          </select>
          <button className="btn btn-ghost btn-sm" onClick={refreshPorts}>Refresh</button>
        </div>
        <div className="config-group">
          <span className="config-label">Baud Rate</span>
          <select className="input" style={{ width: 110 }} value={baud} onChange={e => setBaud(e.target.value)}>
            {['9600','19200','38400','57600','115200','230400'].map(b => <option key={b}>{b}</option>)}
          </select>
        </div>
        <button className="btn btn-success" onClick={connect} disabled={connected || connecting}>
          {connecting ? 'Connecting…' : 'Connect'}
        </button>
        <button className="btn btn-danger" onClick={disconnect} disabled={!connected}>
          Disconnect
        </button>
      </div>

      {/* Main area */}
      <div className="split-pane split-3-1" style={{ flex: 1, minHeight: 0, marginBottom: 10 }}>
        {/* Terminal */}
        <div className="terminal" style={{ height: '100%' }}>
          <div className="terminal-header">
            ■ SERIAL TERMINAL
            <button
              style={{ background: 'none', border: 'none', color: 'var(--text3)', cursor: 'pointer', fontSize: 11 }}
              onClick={() => {/* clear handled by local state not possible since messages come from hook */}}
            >
              CLEAR
            </button>
          </div>
          <div className="terminal-body" ref={termRef}>
            <div className="terminal-line info">[{ts()}] LoRaShield serial terminal ready.</div>
            <div className="terminal-line info">[{ts()}] Select COM port and baud rate, then press CONNECT.</div>
            {messages.map((m, i) => (
              <div key={i} className={`terminal-line ${m.type}`}>
                <span>[{m.ts}] </span>
                {m.type === 'tx' && (
                  <>
                    {m.plain && <span>[TX PLAIN ] {m.plain}</span>}
                    {m.cipher && <><br/><span style={{ color: 'var(--warning)' }}>[TX CIPHER] {m.cipher}</span></>}
                  </>
                )}
                {m.type === 'rx' && (
                  <>
                    {m.cipher && <span style={{ color: 'var(--warning)' }}>[RX CIPHER] {m.cipher}</span>}
                    <br/><span>[RX PLAIN ] {m.plain}</span>
                  </>
                )}
                {m.type === 'error' && <span>[ERR] {m.message}</span>}
              </div>
            ))}
          </div>
        </div>

        {/* Connection log */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
          <p className="card-label">Connection Log</p>
          <div style={{ flex: 1, overflowY: 'auto', fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text2)' }}>
            {connLog.map((l, i) => (
              <div key={i} style={{ marginBottom: 2 }}>[{l.t}] {l.msg}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Send row */}
      <div className="send-row">
        <input
          className="input grow"
          placeholder="Type message to send…"
          value={sendText}
          onChange={e => setSendText(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button className="btn btn-primary" onClick={send} disabled={!connected}>Send</button>
      </div>
    </div>
  )
}
