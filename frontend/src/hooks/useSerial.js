import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = 'ws://127.0.0.1:8000/ws/serial'

let _sharedWs = null
let _listeners = []
let _connState = { connected: false, port: '' }

function getWs() {
  if (_sharedWs && _sharedWs.readyState < 2) return _sharedWs
  _sharedWs = new WebSocket(WS_URL)
  _sharedWs.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data)
      if (data.type === 'status') {
        _connState = { connected: data.connected, port: data.port }
      }
      _listeners.forEach(fn => fn(data))
    } catch {}
  }
  _sharedWs.onclose = () => {
    setTimeout(getWs, 3000)
  }
  return _sharedWs
}

// Initialise the shared WS
try { getWs() } catch {}

export function useSerial() {
  const [connected, setConnected] = useState(_connState.connected)
  const [port, setPort] = useState(_connState.port)

  useEffect(() => {
    const handler = (data) => {
      if (data.type === 'status') {
        setConnected(data.connected)
        setPort(data.port)
      }
    }
    _listeners.push(handler)
    return () => { _listeners = _listeners.filter(f => f !== handler) }
  }, [])

  return { connected, port }
}

export function useSerialMessages() {
  const [messages, setMessages] = useState([])

  useEffect(() => {
    const handler = (data) => {
      if (data.type === 'tx' || data.type === 'rx' || data.type === 'error' || data.type === 'crc_error') {
        const ts = new Date().toLocaleTimeString('en-GB')
        setMessages(prev => [...prev.slice(-200), { ...data, ts }])
      }
    }
    _listeners.push(handler)
    return () => { _listeners = _listeners.filter(f => f !== handler) }
  }, [])

  return messages
}
