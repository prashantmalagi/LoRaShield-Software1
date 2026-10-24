export default function About() {
  const features = [
    { icon: '~', color: 'var(--accent)',   title: 'LoRa Communication',    desc: 'Long-range wireless via SX1278 + ESP32 with real-time serial terminal' },
    { icon: '→', color: 'var(--success)', title: 'Morse Code Encoding',    desc: 'Standard ITU Morse code encoding for all alphanumeric characters' },
    { icon: '★', color: 'var(--purple)',  title: 'AI Decoding',            desc: 'TensorFlow/Keras seq2seq model for intelligent Morse-to-text decoding' },
    { icon: '#', color: 'var(--warning)', title: 'AES-256 Encryption',     desc: 'Military-grade AES-256 CBC encryption with SHA-256 key derivation' },
    { icon: '☰', color: 'var(--teal)',    title: 'Message History',        desc: 'Persistent JSON-based message history with search and CSV export' },
    { icon: '⚙', color: 'var(--text2)',   title: 'Persistent Settings',    desc: 'Auto-connect, encryption key, and baud rate saved across sessions' },
  ]

  const tech = [
    ['React + Vite',    'Frontend framework'],
    ['FastAPI',         'Python REST/WS API'],
    ['PySerial',        'Serial communication'],
    ['pycryptodome',    'AES-256 encryption'],
    ['TensorFlow/Keras','AI model inference'],
    ['WebSocket',       'Real-time serial RX'],
  ]

  return (
    <div className="page">
      <div className="page-header">
        <span className="page-title">About LoRaShield</span>
      </div>
      <div className="page-divider" />

      {/* Hero */}
      <div className="about-hero">
        <div style={{ position: 'relative', zIndex: 1 }}>
          <div className="about-version">v1.0.0</div>
          <h2 style={{ fontSize: 32, fontWeight: 800, marginBottom: 8 }}>LoRaShield</h2>
          <p style={{ color: 'var(--text2)', fontSize: 15, maxWidth: 500, margin: '0 auto 16px' }}>
            AI-Enhanced Secure Morse Code Communication System using LoRa Technology
          </p>
          <p style={{ color: 'var(--text3)', fontSize: 12 }}>
            Built with React · FastAPI · TensorFlow · AES-256 · LoRa SX1278
          </p>
        </div>
      </div>

      {/* Features */}
      <p className="section-label">Features</p>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginBottom: 24 }}>
        {features.map((f, i) => (
          <div className="card" key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
            <div style={{
              width: 36, height: 36, borderRadius: 8, background: `${f.color}18`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              color: f.color, fontSize: 16, fontFamily: 'var(--font-mono)', flexShrink: 0
            }}>{f.icon}</div>
            <div>
              <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>{f.title}</div>
              <div style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.5 }}>{f.desc}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Tech stack */}
      <p className="section-label">Technology Stack</p>
      <div className="card2">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
          {tech.map(([name, desc], i) => (
            <div key={i} style={{ padding: '8px 12px', background: 'var(--card)', borderRadius: 6 }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent)', fontWeight: 600 }}>{name}</div>
              <div style={{ fontSize: 10, color: 'var(--text3)', marginTop: 2 }}>{desc}</div>
            </div>
          ))}
        </div>
      </div>

      <p style={{ color: 'var(--text3)', fontSize: 11, marginTop: 20, textAlign: 'center' }}>
        © 2026 LoRaShield · AI-Enhanced Secure Communication
      </p>
    </div>
  )
}
