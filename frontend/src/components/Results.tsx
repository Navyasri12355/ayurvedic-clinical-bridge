import React from 'react'

export default function Results({ results, loading, error }: any) {
  if (loading) return <div className="results">Loading results…</div>
  if (error) return <div className="results error">Error: {error}</div>
  if (!results) return <div className="results">No results yet. Run a query above.</div>

  return (
    <div className="results">
      <h2>Results</h2>
      <pre>{JSON.stringify(results, null, 2)}</pre>
    </div>
  )
}
