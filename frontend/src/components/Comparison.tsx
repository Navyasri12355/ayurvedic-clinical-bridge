import React, { useState, useEffect } from 'react'

export default function Comparison() {
  const [running, setRunning] = useState(false)
  const [runId, setRunId] = useState<string | null>(null)
  const [status, setStatus] = useState<string | null>(null)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const backend = (import.meta.env.VITE_BACKEND_URL as string) || ''
  const api = (path: string) => (backend ? `${backend}${path}` : path)

  const startRun = async () => {
    setError(null)
    setResults(null)
    setRunning(true)
    setStatus('queued')
    try {
      const res = await fetch(api('/analysis/sequence-tagging'), { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ model: 'both', epochs: 2, tiny: true }) })
      if (!res.ok) throw new Error(await res.text())
      const js = await res.json()
      setRunId(js.run_id)
      setStatus('queued')
    } catch (e: any) {
      setError(e.message || 'Failed to start run')
      setRunning(false)
    }
  }

  useEffect(() => {
    let interval: any
    if (runId && running) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(api('/analysis/sequence-tagging/runs'))
          if (!res.ok) throw new Error(await res.text())
          const list = await res.json()
          const me = list.find((r: any) => r.run_id === runId)
          if (me) {
            setStatus(me.status)
            if (me.status === 'completed') {
              // fetch results
              const r = await fetch(api('/analysis/sequence-tagging'))
              const data = await r.json()
              setResults(data)
              setRunning(false)
              clearInterval(interval)
            }
            if (me.status === 'failed') {
              setError(me.error || 'Run failed')
              setRunning(false)
              clearInterval(interval)
            }
          }
        } catch (e: any) {
          setError(e.message || 'Polling failed')
          setRunning(false)
          clearInterval(interval)
        }
      }, 2000)
    }
    return () => clearInterval(interval)
  }, [runId, running])

  return (
    <div style={{marginTop:16}}>
      <h3>Sequence-tagging comparison ✅</h3>
      <div>
        <button onClick={startRun} disabled={running}>Run BiLSTM vs Transformer (tiny)</button>
        {runId && <span style={{marginLeft:12}}>Run: <strong>{runId}</strong> — Status: {status}</span>}
      </div>
      {error && <div style={{color:'crimson',marginTop:8}}>Error: {error}</div>}
      {results && (
        <div style={{marginTop:12}}>
          <h4>Results</h4>
          <pre style={{background:'#f6f8fa',padding:12,borderRadius:6}}>{JSON.stringify(results, null, 2)}</pre>
          {results.results && (
            <table style={{marginTop:8,borderCollapse:'collapse'}}>
              <thead>
                <tr><th style={{border:'1px solid #ddd',padding:6}}>Model</th><th style={{border:'1px solid #ddd',padding:6}}>F1</th><th style={{border:'1px solid #ddd',padding:6}}>Precision</th><th style={{border:'1px solid #ddd',padding:6}}>Recall</th></tr>
              </thead>
              <tbody>
                {results.results.map((r: any) => (
                  <tr key={r.model}>
                    <td style={{border:'1px solid #ddd',padding:6}}>{r.model}</td>
                    <td style={{border:'1px solid #ddd',padding:6}}>{r.f1.toFixed(3)}</td>
                    <td style={{border:'1px solid #ddd',padding:6}}>{r.precision.toFixed(3)}</td>
                    <td style={{border:'1px solid #ddd',padding:6}}>{r.recall.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  )
}
