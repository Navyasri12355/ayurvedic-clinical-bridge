import React, { useState } from 'react'

interface QueryResult {
  concepts?: any[]
  cross_domain_mappings?: any[]
  confidence_score?: number
  warnings?: string[]
  metadata?: any
}

export default function GeneralUsers() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<QueryResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'}/api/knowledge/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query_text: query,
          query_type: 'general',
          user_role: 'general'
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="general-users-page">
      <div className="header">
        <h1>Ayurvedic Clinical Bridge - General Users</h1>
        <div className="disclaimer">
          <h3>⚠️ Important Disclaimer</h3>
          <p>
            This system provides educational information only and is not intended for medical diagnosis or treatment.
            Always consult with qualified healthcare professionals before making any medical decisions.
            The information provided should not replace professional medical advice.
          </p>
        </div>
      </div>

      <div className="query-section">
        <form onSubmit={handleSubmit} className="query-form">
          <div className="form-group">
            <label htmlFor="query">Ask about Ayurvedic concepts, herbs, or general health information:</label>
            <textarea
              id="query"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., What are the benefits of turmeric? What is Ayurvedic approach to digestive health?"
              rows={3}
              disabled={loading}
            />
          </div>
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {error && (
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}

      {results && (
        <div className="results-section">
          <h2>Educational Information</h2>
          
          {results.warnings && results.warnings.length > 0 && (
            <div className="warnings">
              <h3>⚠️ Important Notes</h3>
              {results.warnings.map((warning, index) => (
                <p key={index}>{warning}</p>
              ))}
            </div>
          )}

          {results.concepts && results.concepts.length > 0 && (
            <div className="concepts">
              <h3>Related Concepts</h3>
              {results.concepts.map((concept, index) => (
                <div key={index} className="concept-card">
                  <h4>{concept.concept_name}</h4>
                  <p><strong>Type:</strong> {concept.concept_type}</p>
                  {concept.descriptions && concept.descriptions.length > 0 && (
                    <div>
                      <strong>Description:</strong>
                      {concept.descriptions.map((desc: string, i: number) => (
                        <p key={i}>{desc}</p>
                      ))}
                    </div>
                  )}
                  {concept.ayurvedic_terms && concept.ayurvedic_terms.length > 0 && (
                    <p><strong>Ayurvedic Terms:</strong> {concept.ayurvedic_terms.join(', ')}</p>
                  )}
                  {concept.sources && concept.sources.length > 0 && (
                    <p><strong>Sources:</strong> {concept.sources.join(', ')}</p>
                  )}
                </div>
              ))}
            </div>
          )}

          {results.cross_domain_mappings && results.cross_domain_mappings.length > 0 && (
            <div className="mappings">
              <h3>Cross-Domain Connections</h3>
              {results.cross_domain_mappings.map((mapping, index) => (
                <div key={index} className="mapping-card">
                  <p><strong>Modern Term:</strong> {mapping.biomedical_concept}</p>
                  <p><strong>Ayurvedic Term:</strong> {mapping.ayurvedic_concept}</p>
                  <p><strong>Confidence:</strong> {(mapping.confidence_score * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          )}

          {results.confidence_score !== undefined && (
            <div className="confidence">
              <p><strong>Information Confidence:</strong> {(results.confidence_score * 100).toFixed(1)}%</p>
            </div>
          )}

          <div className="safety-reminder">
            <h3>🔒 Safety Reminder</h3>
            <p>
              This information is for educational purposes only. Before using any herbs or treatments:
            </p>
            <ul>
              <li>Consult with a qualified Ayurvedic practitioner or healthcare provider</li>
              <li>Inform your doctor about any herbs or supplements you plan to take</li>
              <li>Be aware of potential interactions with medications</li>
              <li>Start with small amounts and monitor for adverse reactions</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}