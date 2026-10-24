import { useState, useEffect } from 'react'

const API = 'http://127.0.0.1:8000'

export default function History() {
  const [records, setRecords] = useState([])
  const [query, setQuery] = useState('')
  const [status, setStatus] = useState('')
  const [loading, setLoading] = useState(false)

  const load = async (q = '') => {
    setLoading(true)
    try {
      const url = q ? `${API}/api/history?q=${encodeURIComponent(q)}` : `${API}/api/history`
      const r = await fetch(url)
      const d = await r.json()
      setRecords(d.records || [])
    } catch {}
    setLoading(false)
  }

  useEffect(() => { load() }, [])

  const del = async (i) => {
    if (!confirm('Delete this record?')) return
    await fetch(`${API}/api/history/${i}`, { method: 'DELETE' })
    setStatus(`Record ${i} deleted.`)
    load(query)
  }

  const clearAll = async () => {
    if (!confirm('Delete all history records?')) return
    await fetch(`${API}/api/history/clear`, { method: 'DELETE' })
    setStatus('History cleared.')
    load()
  }

  const exportCSV = async () => {
    const r = await fetch(`${API}/api/history/export`)
    const d = await r.json()
    setStatus(d.message)
  }

  const truncate = (s, n = 40) => s && s.length > n ? s.slice(0, n) + '…' : s || ''

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-title">Message History</span>
        <span className="text-muted text-sm">{records.length} records</span>
      </div>
      <div className="page-divider" />

      {/* Search + actions */}
      <div className="flex gap-2 items-center mb-4">
        <input
          className="input grow"
          placeholder="Search records…"
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && load(query)}
        />
        <button className="btn btn-primary btn-sm" onClick={() => load(query)}>Search</button>
        <button className="btn btn-ghost btn-sm" onClick={() => { setQuery(''); load('') }}>Reset</button>
        <button className="btn btn-ghost btn-sm" onClick={() => load(query)}>Refresh</button>
        <button className="btn btn-teal btn-sm" onClick={exportCSV}>Export CSV</button>
        <button className="btn btn-danger btn-sm" onClick={clearAll}>Clear All</button>
      </div>
      {status && <p className="text-muted text-sm mb-3">{status}</p>}

      {/* Table */}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Time</th>
              <th>Dir</th>
              <th>Plain Text</th>
              <th>Morse</th>
              <th>Encryption</th>
              <th>Sender</th>
              <th>Receiver</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td colSpan={9} style={{ textAlign: 'center', color: 'var(--text3)', padding: 20 }}>Loading…</td></tr>
            )}
            {!loading && records.length === 0 && (
              <tr><td colSpan={9} style={{ textAlign: 'center', color: 'var(--text3)', padding: 20 }}>No records found.</td></tr>
            )}
            {records.map((rec, i) => (
              <tr key={i}>
                <td className="text-muted">{i}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 10, whiteSpace: 'nowrap' }}>{rec.time}</td>
                <td>
                  <span className={`badge ${rec.direction === 'TX' ? 'badge-tx' : 'badge-rx'}`}>
                    {rec.direction}
                  </span>
                </td>
                <td>{truncate(rec.plain_text)}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}>{truncate(rec.morse_code, 24)}</td>
                <td>
                  <span className={`badge ${rec.encryption_status && rec.encryption_status !== 'None' ? 'badge-enc' : 'badge-none'}`}>
                    {rec.encryption_status || 'None'}
                  </span>
                </td>
                <td>{rec.sender}</td>
                <td>{rec.receiver}</td>
                <td>
                  <button className="btn btn-danger btn-sm" onClick={() => del(i)}>✕</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
