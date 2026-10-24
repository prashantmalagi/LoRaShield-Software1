/**
 * LoRaShield – Morse to Text (AI) Decoder Page
 * =============================================
 * Features:
 *   1. Compact real-time AI Model status indicator near page header.
 *   2. Live Morse character prefix suggestions (deterministic ITU Morse matching).
 *   3. Interactive Morse Code typing error correction (Levenshtein distance).
 *   4. Collapsible correction settings to minimize vertical clutter.
 *   5. AI confidence meter shown only when a genuine AI prediction is made.
 *   6. BiLSTM sequence decoder integration with dictionary fallback.
 */

import { useState, useCallback, useEffect } from 'react'

const API = 'http://127.0.0.1:8000'

// ─── Colour tokens (mirrors index.css variables) ───────────────────────────
const CLR = {
  success: 'var(--success)',
  warning: 'var(--warning)',
  error:   'var(--error)',
  accent:  'var(--accent)',
  text1:   'var(--text1)',
  text2:   'var(--text2)',
  text3:   'var(--text3)',
  card2:   'var(--card2)',
  border:  'var(--border)',
  border2: 'var(--border2)',
}

// ─── Standard ITU Morse Character Dictionary ────────────────────────────────
// Strictly mirrors utils.morse_correction.MORSE_DICT for 100% compatibility.
const ITU_MORSE_TABLE = [
  // Letters
  { char: 'A', morse: '.-' },    { char: 'B', morse: '-...' },  { char: 'C', morse: '-.-.' },
  { char: 'D', morse: '-..' },   { char: 'E', morse: '.' },     { char: 'F', morse: '..-.' },
  { char: 'G', morse: '--.' },   { char: 'H', morse: '....' },  { char: 'I', morse: '..' },
  { char: 'J', morse: '.---' },  { char: 'K', morse: '-.-' },   { char: 'L', morse: '.-..' },
  { char: 'M', morse: '--' },    { char: 'N', morse: '-.' },    { char: 'O', morse: '---' },
  { char: 'P', morse: '.--.' },  { char: 'Q', morse: '--.-' },  { char: 'R', morse: '.-.' },
  { char: 'S', morse: '...' },   { char: 'T', morse: '-' },     { char: 'U', morse: '..-' },
  { char: 'V', morse: '...-' },  { char: 'W', morse: '.--' },   { char: 'X', morse: '-..-' },
  { char: 'Y', morse: '-.--' },  { char: 'Z', morse: '--..' },
  // Digits
  { char: '0', morse: '-----' }, { char: '1', morse: '.----' }, { char: '2', morse: '..---' },
  { char: '3', morse: '...--' }, { char: '4', morse: '....-' }, { char: '5', morse: '.....' },
  { char: '6', morse: '-....' }, { char: '7', morse: '--...' }, { char: '8', morse: '---..' },
  { char: '9', morse: '----.' },
  // Common Punctuation
  { char: '.', morse: '.-.-.-' }, { char: ',', morse: '--..--' }, { char: '?', morse: '..--..' },
  { char: "'", morse: '.----.' }, { char: '!', morse: '-.-.--' }, { char: '/', morse: '-..-.' },
  { char: '(', morse: '-.--.' },  { char: ')', morse: '-.--.-' }, { char: '&', morse: '.-...' },
  { char: ':', morse: '---...' }, { char: ';', morse: '-.-.-.' }, { char: '=', morse: '-...-' },
  { char: '+', morse: '.-.-.' },  { char: '-', morse: '-....-' }, { char: '"', morse: '.-..-.' },
  { char: '$', morse: '...-..-' },{ char: '@', morse: '.--.-.' },
]

/**
 * Extract the current unfinished Morse token from the input string.
 *
 * Rules:
 *   - If the input ends with whitespace or a slash '/', the symbol has been
 *     completed or committed; returns '' (no active prefix).
 *   - Otherwise extracts the trailing contiguous dot and dash sequence.
 */
function extractCurrentPrefix(morseValue) {
  if (!morseValue) return ''
  if (/[\s/]/.test(morseValue.slice(-1))) return ''
  const match = morseValue.match(/[.-]+$/)
  return match ? match[0] : ''
}

/**
 * Deterministic prefix matching over the ITU Morse table.
 *
 * Sorts:
 *   1. Exact complete code matches first (e.g. E for '.', A for '.-', T for '-')
 *   2. Shorter Morse code length ascending
 *   3. Alphabetically
 */
function getMorsePrefixCandidates(prefix) {
  if (!prefix) return []
  const cleanPrefix = prefix.trim()
  if (!cleanPrefix || !/^[.-]+$/.test(cleanPrefix)) return []

  const results = []
  for (let i = 0; i < ITU_MORSE_TABLE.length; i++) {
    const entry = ITU_MORSE_TABLE[i]
    if (entry.morse.startsWith(cleanPrefix)) {
      results.push({
        char: entry.char,
        morse: entry.morse,
        exact: entry.morse === cleanPrefix,
      })
    }
  }

  results.sort((a, b) => {
    if (a.exact !== b.exact) return a.exact ? -1 : 1
    if (a.morse.length !== b.morse.length) return a.morse.length - b.morse.length
    return a.char.localeCompare(b.char)
  })

  return results
}

// ─── Inline Components ─────────────────────────────────────────────────────

/**
 * Chip for a live Morse character suggestion.
 */
function PrefixChip({ char, morse, exact, onSelect }) {
  return (
    <button
      type="button"
      id={`prefix-chip-${char}`}
      onClick={() => onSelect(char, morse)}
      title={`Select ${char} (${morse})${exact ? ' — exact complete code' : ''}`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '3px 8px',
        border: `1px solid ${exact ? CLR.success : CLR.border2}`,
        borderRadius: 5,
        background: exact ? 'rgba(63,185,80,0.1)' : 'var(--card2)',
        color: exact ? CLR.success : CLR.text1,
        fontFamily: 'var(--font-mono)',
        fontSize: 12,
        cursor: 'pointer',
        transition: 'all 0.12s ease',
        flexShrink: 0,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.borderColor = exact ? CLR.success : CLR.accent
        e.currentTarget.style.background = exact ? 'rgba(63,185,80,0.22)' : 'rgba(37,99,235,0.2)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.borderColor = exact ? CLR.success : CLR.border2
        e.currentTarget.style.background = exact ? 'rgba(63,185,80,0.1)' : 'var(--card2)'
      }}
    >
      <span style={{ fontWeight: 700, fontSize: 13, color: exact ? CLR.success : CLR.accent }}>
        {char}
      </span>
      <span style={{ fontSize: 11, color: CLR.text2, letterSpacing: 1 }}>
        {morse}
      </span>
      {exact && (
        <span style={{ fontSize: 9, color: CLR.success, fontWeight: 700 }}>
          ✓
        </span>
      )}
    </button>
  )
}

/**
 * Live Morse character suggestions container placed directly beneath Morse input.
 */
function MorsePrefixSuggestions({ prefix, suggestions, onSelect }) {
  if (!prefix) return null

  return (
    <div
      id="morse-prefix-suggestions"
      style={{
        marginTop: 8,
        padding: '8px 12px',
        borderRadius: 6,
        border: `1px solid ${CLR.border}`,
        background: 'rgba(22, 27, 34, 0.95)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: 1, color: CLR.text3, textTransform: 'uppercase' }}>
          Live Character Suggestions
        </span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
            fontWeight: 700,
            color: CLR.accent,
            padding: '1px 6px',
            border: `1px solid ${CLR.accent}`,
            borderRadius: 4,
            letterSpacing: 1.5,
          }}
        >
          {prefix}
        </span>
        <span style={{ fontSize: 10, color: CLR.text3, marginLeft: 'auto' }}>
          {suggestions.length === 0
            ? 'No matches'
            : `${suggestions.length} candidate${suggestions.length > 1 ? 's' : ''} (click to complete)`}
        </span>
      </div>

      {suggestions.length > 0 ? (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: 5,
            maxHeight: 120,
            overflowY: 'auto',
            paddingRight: 2,
          }}
        >
          {suggestions.map(s => (
            <PrefixChip
              key={`${s.char}-${s.morse}`}
              char={s.char}
              morse={s.morse}
              exact={s.exact}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : (
        <p style={{ margin: 0, fontSize: 11, color: CLR.error, opacity: 0.9 }}>
          No standard Morse character begins with <code style={{ fontFamily: 'var(--font-mono)' }}>{prefix}</code>.
        </p>
      )}
    </div>
  )
}

/**
 * Suggestion chip for Check & Correct typing error suggestions.
 */
function SuggestionChip({ char, morse, score, wordIndex, tokenIndex, onSelect }) {
  const pct = Math.round(score * 100)
  return (
    <button
      type="button"
      id={`suggestion-${wordIndex}-${tokenIndex}-${char}`}
      onClick={() => onSelect(char, morse, wordIndex, tokenIndex)}
      title={`Select ${char} (${morse}) – ${pct}% similarity`}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 6,
        padding: '4px 9px',
        border: `1px solid ${CLR.accent}`,
        borderRadius: 5,
        background: 'transparent',
        color: CLR.accent,
        fontFamily: 'var(--font-mono)',
        fontSize: 12,
        cursor: 'pointer',
        transition: 'background 0.15s, color 0.15s',
        marginRight: 6,
        marginBottom: 4,
      }}
      onMouseEnter={e => {
        e.currentTarget.style.background = CLR.accent
        e.currentTarget.style.color = '#fff'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.color = CLR.accent
      }}
    >
      <strong>{char}</strong>
      <span style={{ opacity: 0.8 }}>({morse})</span>
      <span style={{ fontSize: 10, opacity: 0.7 }}>{pct}%</span>
    </button>
  )
}

/**
 * Row displaying an invalid Morse token and possible corrections.
 */
function CorrectionRow({ token, onSelect }) {
  if (token.is_valid) return null

  const { input, suggestions, reliable, word_index, token_index } = token

  return (
    <div
      id={`correction-row-${word_index}-${token_index}`}
      style={{
        padding: '8px 12px',
        marginBottom: 6,
        borderRadius: 6,
        border: `1px solid ${reliable ? CLR.warning : CLR.error}`,
        background: reliable ? 'rgba(210,153,34,0.06)' : 'rgba(248,81,73,0.06)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: suggestions.length > 0 ? 6 : 0 }}>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 14,
            fontWeight: 700,
            color: reliable ? CLR.warning : CLR.error,
            letterSpacing: 2,
          }}
        >
          {input}
        </span>
        <span style={{ fontSize: 11, color: CLR.text2 }}>
          {reliable ? '– Unknown Morse code. Did you mean?' : '– Pattern too different from any known Morse character.'}
        </span>
      </div>

      {reliable && suggestions.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
          {suggestions.map(s => (
            <SuggestionChip
              key={`${s.char}-${s.morse}`}
              char={s.char}
              morse={s.morse}
              score={s.score}
              wordIndex={word_index}
              tokenIndex={token_index}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Check & Correct result panel.
 */
function CorrectionPanel({ corrections, onSelect, onDismiss }) {
  if (!corrections || !corrections.tokens || corrections.tokens.length === 0) return null

  const invalidTokens = corrections.tokens.filter(t => !t.is_valid)

  if (invalidTokens.length === 0) {
    return (
      <div
        id="correction-panel-success"
        style={{
          padding: '8px 14px',
          borderRadius: 6,
          border: `1px solid ${CLR.success}`,
          background: 'rgba(63,185,80,0.07)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 12,
        }}
      >
        <span style={{ color: CLR.success, fontSize: 12, fontWeight: 600 }}>
          ✓ All Morse patterns are valid ITU codes — ready to decode.
        </span>
        <button
          type="button"
          id="correction-dismiss-btn"
          onClick={onDismiss}
          style={{ background: 'none', border: 'none', color: CLR.text2, cursor: 'pointer', fontSize: 16 }}
        >
          ×
        </button>
      </div>
    )
  }

  return (
    <div
      id="correction-panel"
      style={{
        marginBottom: 12,
        padding: 12,
        borderRadius: 8,
        border: `1px solid ${CLR.border}`,
        background: CLR.card2,
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 700, color: CLR.text2, letterSpacing: 1 }}>
          MORSE CORRECTION SUGGESTIONS
          <span style={{ marginLeft: 8, color: CLR.warning }}>
            ({invalidTokens.length} invalid token{invalidTokens.length > 1 ? 's' : ''})
          </span>
        </span>
        <button
          type="button"
          id="correction-panel-close"
          onClick={onDismiss}
          title="Close correction panel"
          style={{ background: 'none', border: 'none', color: CLR.text2, cursor: 'pointer', fontSize: 16 }}
        >
          ×
        </button>
      </div>

      {invalidTokens.map((tok, i) => (
        <CorrectionRow
          key={`${tok.word_index}-${tok.token_index}-${i}`}
          token={tok}
          onSelect={onSelect}
        />
      ))}

      <p style={{ margin: '4px 0 0', fontSize: 10, color: CLR.text3 }}>
        Click a suggested character above to replace the token in the input field.
      </p>
    </div>
  )
}

// ─── Main Decoder Page ──────────────────────────────────────────────────────

export default function Decoder() {
  const [morse, setMorse]         = useState('')
  const [output, setOutput]       = useState('')
  const [status, setStatus]       = useState('Ready.')
  const [loading, setLoading]     = useState(false)

  // AI model status
  const [aiLoaded, setAiLoaded]   = useState(false)
  const [aiLoading, setAiLoading] = useState(false)
  const [aiError, setAiError]     = useState(null)

  // Prediction outcome (null until decode is called)
  const [prediction, setPrediction] = useState(null)

  // Correction state
  const [corrections, setCorrections]       = useState(null)
  const [correcting, setCorrecting]         = useState(false)
  const [threshold, setThreshold]           = useState(0.65)
  const [showThresholdCtrl, setShowThreshold] = useState(false)

  // Live prefix suggestion state
  const [currentPrefix, setCurrentPrefix]         = useState('')
  const [prefixSuggestions, setPrefixSuggestions] = useState([])

  // Query actual backend AI model status on mount
  useEffect(() => {
    let active = true
    const checkModel = async () => {
      try {
        const r = await fetch(`${API}/api/ai/status`)
        if (r.ok && active) {
          const data = await r.json()
          setAiLoaded(Boolean(data.loaded))
        }
      } catch {
        // Backend offline or starting up
      }
    }
    checkModel()
    return () => { active = false }
  }, [])

  // ─── Input & Suggestion Handlers ──────────────────────────────────────────

  const handleMorseChange = (e) => {
    const val = e.target.value
    setMorse(val)
    const prefix = extractCurrentPrefix(val)
    setCurrentPrefix(prefix)
    const candidates = getMorsePrefixCandidates(prefix)
    setPrefixSuggestions(candidates)
    setCorrections(null)
  }

  const handlePrefixSelect = useCallback((char, suggMorse) => {
    setMorse(prev => {
      const match = prev.match(/[.-]+$/)
      if (!match) {
        return (prev ? prev.trimEnd() + ' ' : '') + suggMorse + ' '
      }
      const prefixLen = match[0].length
      return prev.slice(0, prev.length - prefixLen) + suggMorse + ' '
    })
    setCurrentPrefix('')
    setPrefixSuggestions([])
    setCorrections(null)
    setStatus(`Completed character: ${char} (${suggMorse}).`)
  }, [])

  // ─── AI Model Loading ─────────────────────────────────────────────────────

  const loadModel = async () => {
    setAiLoading(true)
    setAiError(null)
    setStatus('Loading AI model…')
    try {
      const r = await fetch(`${API}/api/ai/load`, { method: 'POST' })
      const data = await r.json()
      if (data.ok) {
        setAiLoaded(true)
        setAiError(null)
        setStatus('AI model loaded successfully.')
      } else {
        setAiLoaded(false)
        setAiError(data.message || 'Load failed')
        setStatus(`AI Load failed: ${data.message || 'Unknown error'}`)
      }
    } catch (e) {
      setAiLoaded(false)
      setAiError(e.message)
      setStatus(`Error loading AI model: ${e.message}`)
    } finally {
      setAiLoading(false)
    }
  }

  // ─── Decode & Actions ─────────────────────────────────────────────────────

  const decode = async () => {
    if (!morse.trim()) { setStatus('Error: Morse input is empty.'); return }
    setLoading(true)
    setStatus('Decoding Morse code…')
    try {
      const r = await fetch(`${API}/api/morse/decode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ morse }),
      })
      if (!r.ok) throw new Error(`Server returned ${r.status}`)
      const data = await r.json()
      setOutput(data.text)
      setPrediction({
        text: data.text,
        confidence: Number(data.confidence) || 0,
        elapsed_ms: Number(data.elapsed_ms) || 0,
        method: data.method || 'Dictionary',
      })
      setStatus(`Decoded ${morse.length} characters (${data.method}).`)
    } catch (e) {
      setStatus(`Decode error: ${e.message}`)
    } finally {
      setLoading(false)
    }
  }

  const copy = () => {
    if (!output) return
    navigator.clipboard.writeText(output)
    setStatus('Decoded text copied to clipboard.')
  }

  const clear = () => {
    setMorse('')
    setOutput('')
    setPrediction(null)
    setCurrentPrefix('')
    setPrefixSuggestions([])
    setCorrections(null)
    setStatus('Cleared.')
  }

  // ─── Check & Correct ──────────────────────────────────────────────────────

  const checkAndCorrect = useCallback(async () => {
    if (!morse.trim()) { setStatus('Error: Morse input is empty.'); return }
    setCorrecting(true)
    setStatus('Checking Morse patterns against ITU standard…')
    setCorrections(null)
    try {
      const r = await fetch(`${API}/api/morse/correct-message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: morse, threshold }),
      })
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      const data = await r.json()
      setCorrections(data)
      if (data.all_valid) {
        setStatus(`✓ All ${data.token_count} Morse token(s) are valid ITU codes.`)
      } else {
        const n = data.tokens.filter(t => !t.is_valid).length
        setStatus(`Found ${n} invalid Morse token${n > 1 ? 's' : ''}. Select a correction below.`)
      }
    } catch (e) {
      setStatus(`Correction check failed: ${e.message}`)
    } finally {
      setCorrecting(false)
    }
  }, [morse, threshold])

  const handleSuggestionSelect = useCallback(async (char, suggMorse, wordIndex, tokenIndex) => {
    try {
      const r = await fetch(`${API}/api/morse/apply-correction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: morse,
          word_index: wordIndex,
          token_index: tokenIndex,
          replacement_morse: suggMorse,
        }),
      })
      if (!r.ok) throw new Error(`Server error ${r.status}`)
      const data = await r.json()
      const correctedMsg = data.corrected_message

      setMorse(correctedMsg)
      setStatus(`Replaced token with ${char} (${suggMorse}). Click Decode to proceed.`)

      const r2 = await fetch(`${API}/api/morse/correct-message`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: correctedMsg, threshold }),
      })
      if (r2.ok) {
        const data2 = await r2.json()
        setCorrections(data2)
        if (data2.all_valid) {
          setStatus(`✓ All tokens corrected — ready to decode (last: ${char} = ${suggMorse}).`)
        }
      }
    } catch (e) {
      setStatus(`Failed to apply correction: ${e.message}`)
    }
  }, [morse, threshold])

  // ─── Confidence Color Determination ────────────────────────────────────────
  const confVal = prediction?.confidence ?? 0
  const confColor = confVal >= 80 ? CLR.success : confVal >= 60 ? CLR.warning : CLR.error

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      {/* ── Page Header with Compact Model-Status Indicator ─────────────────── */}
      <div className="page-header" style={{ alignItems: 'center', marginBottom: 4 }}>
        <span className="page-title">Morse Code → Text (AI)</span>

        {/* Compact AI Model Status Indicator */}
        <div
          id="ai-model-status"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 8,
            padding: '4px 10px',
            borderRadius: 6,
            background: CLR.card2,
            border: `1px solid ${aiError ? 'rgba(248,81,73,0.5)' : CLR.border2}`,
          }}
          title={aiError ? `Error: ${aiError}` : aiLoaded ? 'BiLSTM neural network is active' : 'AI model is not loaded'}
        >
          <div
            className={`led ${aiLoading ? 'warning led-pulse' : aiLoaded ? 'connected' : 'error'}`}
          />
          <span
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: aiLoading ? CLR.warning : aiLoaded ? CLR.success : CLR.text2,
            }}
          >
            {aiLoading ? 'Loading AI…' : aiLoaded ? 'AI Loaded' : 'Not Loaded'}
          </span>

          {!aiLoaded && !aiLoading && (
            <button
              type="button"
              id="btn-load-ai"
              className="btn btn-purple btn-sm"
              onClick={loadModel}
              disabled={aiLoading}
              style={{ padding: '2px 8px', fontSize: 11, marginLeft: 2 }}
            >
              Load AI Model
            </button>
          )}

          {aiError && (
            <span
              style={{
                fontSize: 10,
                color: CLR.error,
                marginLeft: 2,
                maxWidth: 160,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
              title={aiError}
            >
              ({aiError})
            </span>
          )}
        </div>
      </div>

      <div className="page-divider" style={{ margin: '6px 0 12px' }} />

      <p className="page-subtitle" style={{ marginBottom: 12 }}>
        Real-time Morse character suggestions, offline ITU error correction, and BiLSTM AI decoding.
      </p>

      {/* ── Two-Column Main Content Layout ──────────────────────────────────── */}
      <div className="split-pane mb-3">
        {/* Left Column: Morse Code Input + Live Character Suggestions */}
        <div className="flex-col" style={{ minWidth: 0 }}>
          <div className="flex items-center justify-between mb-2">
            <span className="input-label" style={{ margin: 0 }}>Morse Code Input</span>
            <span className="text-muted text-xs">Letters: [space] · Words: [ / ]</span>
          </div>
          <textarea
            id="morse-input-textarea"
            className="textarea mono"
            style={{ minHeight: 180, resize: 'vertical' }}
            placeholder=". - . - / . . . / ."
            value={morse}
            onChange={handleMorseChange}
            spellCheck="false"
          />
          {/* Live Morse Character Suggestions directly beneath Morse input */}
          <MorsePrefixSuggestions
            prefix={currentPrefix}
            suggestions={prefixSuggestions}
            onSelect={handlePrefixSelect}
          />
        </div>

        {/* Right Column: Decoded Text Output */}
        <div className="flex-col" style={{ minWidth: 0 }}>
          <div className="flex items-center justify-between mb-2">
            <span className="input-label" style={{ margin: 0 }}>Decoded Text Output</span>
            <span className="text-muted text-xs">AI &amp; Dictionary output</span>
          </div>
          <textarea
            id="morse-output-textarea"
            className="textarea"
            style={{ minHeight: 180, resize: 'vertical' }}
            readOnly
            value={output}
            placeholder="Decoded text will appear here…"
          />
        </div>
      </div>

      {/* ── Action Controls Bar ─────────────────────────────────────────────── */}
      <div className="flex gap-2 items-center mb-3" style={{ flexWrap: 'wrap' }}>
        <button
          type="button"
          id="btn-decode"
          className="btn btn-primary"
          onClick={decode}
          disabled={loading}
        >
          {loading ? 'Decoding…' : 'Decode'}
        </button>

        <button
          type="button"
          id="btn-check-correct"
          className="btn"
          onClick={checkAndCorrect}
          disabled={correcting}
          style={{
            background: 'transparent',
            border: `1px solid ${CLR.warning}`,
            color: CLR.warning,
            fontWeight: 700,
            padding: '7px 14px',
            borderRadius: 4,
            cursor: 'pointer',
            fontSize: 12,
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => { e.currentTarget.style.background = 'rgba(210,153,34,0.12)' }}
          onMouseLeave={e => { e.currentTarget.style.background = 'transparent' }}
          title="Check for Morse typing errors against standard ITU codes"
        >
          {correcting ? 'Checking…' : '⚡ Check & Correct'}
        </button>

        <button type="button" id="btn-copy" className="btn btn-ghost" onClick={copy}>Copy</button>
        <button type="button" id="btn-clear" className="btn btn-danger" onClick={clear}>Clear</button>

        <span className="text-muted text-xs" style={{ marginLeft: 'auto' }}>
          {status}
        </span>
      </div>

      {/* ── Collapsible Correction Settings ─────────────────────────────────── */}
      <div
        style={{
          marginBottom: 12,
          borderRadius: 6,
          border: `1px solid ${CLR.border}`,
          background: CLR.card2,
          overflow: 'hidden',
        }}
      >
        <button
          type="button"
          id="btn-toggle-correction-settings"
          onClick={() => setShowThreshold(v => !v)}
          style={{
            width: '100%',
            background: 'none',
            border: 'none',
            padding: '7px 12px',
            textAlign: 'left',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: CLR.text2,
            fontSize: 11,
            fontWeight: 700,
            letterSpacing: 1,
          }}
        >
          <span style={{ fontSize: 9 }}>{showThresholdCtrl ? '▼' : '▶'}</span>
          <span>CORRECTION SETTINGS</span>
          <span style={{ marginLeft: 'auto', fontWeight: 500, letterSpacing: 0, color: CLR.text3 }}>
            Threshold: {Math.round(threshold * 100)}%
          </span>
        </button>

        {showThresholdCtrl && (
          <div
            style={{
              padding: '6px 14px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              borderTop: `1px solid ${CLR.border}`,
            }}
          >
            <span style={{ fontSize: 11, color: CLR.text2, minWidth: 130 }}>
              Similarity threshold
            </span>
            <input
              id="correction-threshold-slider"
              type="range"
              min={50}
              max={95}
              step={5}
              value={Math.round(threshold * 100)}
              onChange={e => setThreshold(Number(e.target.value) / 100)}
              style={{ flex: 1 }}
            />
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: CLR.accent, minWidth: 38 }}>
              {Math.round(threshold * 100)}%
            </span>
            <span style={{ fontSize: 10, color: CLR.text3 }}>
              ↑ stricter · ↓ more permissive
            </span>
          </div>
        )}
      </div>

      {/* ── Interactive Morse Error Correction Panel ────────────────────────── */}
      <CorrectionPanel
        corrections={corrections}
        onSelect={handleSuggestionSelect}
        onDismiss={() => setCorrections(null)}
      />

      {/* ── AI Confidence Section ───────────────────────────────────────────── */}
      {/* Hidden until a valid prediction has been produced. */}
      {prediction && (
        prediction.method === 'AI' && prediction.confidence > 0 ? (
          <div id="ai-confidence-section" className="card2 mb-3">
            <div className="flex items-center gap-4">
              <span className="config-label" style={{ margin: 0 }}>AI Confidence</span>
              <span style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-mono)', color: confColor }}>
                {prediction.confidence.toFixed(1)}%
              </span>
              <span className="text-muted text-xs" style={{ marginLeft: 'auto' }}>
                BiLSTM Model (v4) · {prediction.elapsed_ms.toFixed(1)} ms
              </span>
            </div>
            <div className="confidence-bar-track">
              <div
                className="confidence-bar-fill"
                style={{
                  width: `${Math.min(100, Math.max(0, prediction.confidence))}%`,
                  background: confColor,
                }}
              />
            </div>
          </div>
        ) : (
          <div id="dictionary-decode-notice" className="card2 mb-3" style={{ padding: '8px 14px' }}>
            <div className="flex items-center justify-between">
              <span style={{ fontSize: 12, color: CLR.text2 }}>
                Decoded using <strong style={{ color: CLR.text1 }}>ITU Standard Dictionary</strong> (Deterministic fallback)
              </span>
              <span className="text-muted text-xs">
                {prediction.elapsed_ms.toFixed(1)} ms
              </span>
            </div>
          </div>
        )
      )}
    </div>
  )
}
