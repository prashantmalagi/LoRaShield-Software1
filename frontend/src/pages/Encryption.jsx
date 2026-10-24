import { useState, useEffect } from 'react'

const API = 'http://127.0.0.1:8000'

export default function Encryption() {
  const [key, setKey] = useState('LoRaShield2026')
  const [showKey, setShowKey] = useState(false)
  const [plaintext, setPlaintext] = useState('')
  const [encrypted, setEncrypted] = useState('')
  const [decrypted, setDecrypted] = useState('')
  const [status, setStatus] = useState('Ready.')
  const [cryptoOk, setCryptoOk] = useState(true)

  useEffect(() => {
    fetch(`${API}/api/encrypt/status`).then(r => r.json()).then(d => setCryptoOk(d.available)).catch(() => {})
    fetch(`${API}/api/settings`).then(r => r.json()).then(d => { if (d.enc_key) setKey(d.enc_key) }).catch(() => {})
  }, [])

  const generateKey = async () => {
    const r = await fetch(`${API}/api/encrypt/generate-key`, { method: 'POST' })
    const d = await r.json()
    setKey(d.key)
    setShowKey(true)
    setStatus('New 32-character key generated. Save it securely.')
  }

  const copyKey = () => {
    navigator.clipboard.writeText(key)
    setStatus('Key copied to clipboard.')
  }

  const encrypt = async () => {
    if (!key) { setStatus('Error: Key is required.'); return }
    if (!plaintext) { setStatus('Error: Plaintext is empty.'); return }
    try {
      const r = await fetch(`${API}/api/encrypt/encrypt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: plaintext, key }),
      })
      const d = await r.json()
      if (!r.ok) { setStatus(`Error: ${d.detail}`); return }
      setEncrypted(d.encrypted)
      setStatus('Encrypted successfully with AES-256 CBC.')
    } catch (e) { setStatus(`Error: ${e.message}`) }
  }

  const decrypt = async () => {
    if (!key) { setStatus('Error: Key is required.'); return }
    if (!plaintext) { setStatus('Error: Paste ciphertext in input to decrypt.'); return }
    try {
      const r = await fetch(`${API}/api/encrypt/decrypt`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ encrypted: plaintext, key }),
      })
      const d = await r.json()
      if (!r.ok) { setStatus(`Error: ${d.detail}`); return }
      setDecrypted(d.plaintext)
      setStatus('Decrypted successfully.')
    } catch (e) { setStatus(`Error: ${e.message}`) }
  }

  const clearAll = () => {
    setPlaintext(''); setEncrypted(''); setDecrypted(''); setStatus('Cleared.')
  }

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-title">AES-256 Encryption</span>
        <span style={{ fontSize: 12, color: cryptoOk ? 'var(--success)' : 'var(--error)' }}>
          {cryptoOk ? '● pycryptodome Ready' : '● pycryptodome Not Installed'}
        </span>
      </div>
      <div className="page-divider" />

      {/* Key field */}
      <p className="section-label" style={{ marginTop: 0 }}>Secret Key</p>
      <div className="card2 mb-3">
        <div className="input-row">
          <input
            className="input input-mono grow"
            type={showKey ? 'text' : 'password'}
            value={key}
            onChange={e => setKey(e.target.value)}
            placeholder="Enter encryption key…"
          />
          <button className="btn btn-ghost btn-sm" onClick={() => setShowKey(v => !v)}>
            {showKey ? 'Hide' : 'Show'}
          </button>
        </div>
        <div className="flex gap-2 mt-2">
          <button className="btn btn-teal btn-sm" onClick={generateKey}>Generate Key</button>
          <button className="btn btn-ghost btn-sm" onClick={copyKey}>Copy Key</button>
        </div>
        <p className="text-muted text-xs mt-2">Key is SHA-256 hashed to 256 bits internally. Any length accepted.</p>
      </div>

      {/* Plaintext input */}
      <p className="section-label">Plaintext / Ciphertext Input</p>
      <textarea
        className="textarea mb-3"
        style={{ height: 100 }}
        placeholder="Enter message to encrypt, or paste ciphertext to decrypt…"
        value={plaintext}
        onChange={e => setPlaintext(e.target.value)}
      />

      {/* Action buttons */}
      <div className="flex gap-2 items-center mb-4">
        <button className="btn btn-primary" onClick={encrypt} disabled={!cryptoOk}>Encrypt</button>
        <button className="btn btn-teal"    onClick={decrypt} disabled={!cryptoOk}>Decrypt</button>
        <button className="btn btn-danger"  onClick={clearAll}>Clear All</button>
        <span className="text-muted text-sm" style={{ marginLeft: 'auto' }}>{status}</span>
      </div>

      {/* Encrypted output */}
      <div className="flex items-center justify-between mb-2">
        <p className="section-label" style={{ margin: 0 }}>Encrypted Output (AES-256 CBC / Base64)</p>
        <button className="btn btn-ghost btn-sm" onClick={() => { navigator.clipboard.writeText(encrypted); setStatus('Encrypted copied.') }}>Copy</button>
      </div>
      <textarea
        className="textarea mb-4"
        style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--warning)', height: 80 }}
        readOnly
        value={encrypted}
        placeholder="Encrypted output appears here…"
      />

      {/* Decrypted output */}
      <div className="flex items-center justify-between mb-2">
        <p className="section-label" style={{ margin: 0 }}>Decrypted Output</p>
        <button className="btn btn-ghost btn-sm" onClick={() => { navigator.clipboard.writeText(decrypted); setStatus('Decrypted copied.') }}>Copy</button>
      </div>
      <textarea
        className="textarea"
        style={{ color: 'var(--success)', height: 80 }}
        readOnly
        value={decrypted}
        placeholder="Decrypted text appears here…"
      />

      <p className="text-muted text-xs mt-2">
        To decrypt: paste ciphertext in the Plaintext / Ciphertext Input field and press DECRYPT.
      </p>
    </div>
  )
}
