import { useState } from 'react'

const API = 'http://127.0.0.1:8000'

export default function Encoder() {
  const [input, setInput] = useState('')
  const [morse, setMorse] = useState('')
  const [status, setStatus] = useState('Ready.')
  const [loading, setLoading] = useState(false)

  const convert = async () => {
    if (!input.trim()) { setStatus('Error: Input is empty.'); return }
    setLoading(true)
    try {
      const r = await fetch(`${API}/api/morse/encode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      })
      const data = await r.json()
      setMorse(data.morse)
      setStatus(`Converted ${data.input_length} characters.`)
    } catch (e) {
      setStatus(`Error: ${e.message}`)
    }
    setLoading(false)
  }

  const copy = () => {
    if (!morse) { setStatus('Nothing to copy.'); return }
    navigator.clipboard.writeText(morse)
    setStatus('Morse code copied to clipboard.')
  }

  const save = async () => {
    if (!morse) { setStatus('Nothing to save. Convert first.'); return }
    await fetch(`${API}/api/history`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        direction: 'TX', plain_text: input, morse_code: morse,
        encrypted_data: '', sender: 'LOCAL', receiver: 'N/A', encryption_status: 'None',
      }),
    })
    setStatus('Saved to history.')
  }

  const clear = () => {
    setInput(''); setMorse(''); setStatus('Cleared.')
  }

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-title">Text → Morse Code</span>
        <span className="text-muted text-sm">{input.length} chars</span>
      </div>
      <div className="page-divider" />
      <p className="page-subtitle">Convert plain text to standard Morse code.</p>

      <div className="split-pane" style={{ height: 340, marginBottom: 14 }}>
        {/* Input */}
        <div className="flex-col gap-2">
          <p className="input-label">Input Text</p>
          <textarea
            className="textarea grow"
            style={{ resize: 'none', height: '100%' }}
            placeholder="Type your message here…"
            value={input}
            onChange={e => setInput(e.target.value)}
          />
        </div>
        {/* Output */}
        <div className="flex-col gap-2">
          <p className="input-label">Morse Code Output</p>
          <textarea
            className="textarea mono grow"
            style={{ resize: 'none', height: '100%' }}
            readOnly
            placeholder="Morse code will appear here…"
            value={morse}
          />
        </div>
      </div>

      {/* Controls */}
      <div className="flex gap-2 items-center">
        <button className="btn btn-primary" onClick={convert} disabled={loading}>
          {loading ? 'Converting…' : 'Convert'}
        </button>
        <button className="btn btn-ghost" onClick={copy}>Copy</button>
        <button className="btn btn-ghost" onClick={save}>Save to History</button>
        <button className="btn btn-danger" onClick={clear}>Clear</button>
        <span className="text-muted text-sm" style={{ marginLeft: 'auto' }}>{status}</span>
      </div>
    </div>
  )
}
