import React, { useState } from 'react'

type Endpoint = 'api-info' | 'safety-analyze' | 'search-interactions'

interface Props {
  onSubmit: (payload: any) => void
  loading?: boolean
}

export default function QueryForm({ onSubmit, loading }: Props) {
  const [endpoint, setEndpoint] = useState<Endpoint>('api-info')
  const [query, setQuery] = useState<string>('')
  const [herbs, setHerbs] = useState<string>('')
  const [drugs, setDrugs] = useState<string>('')
  const [searchType, setSearchType] = useState<'both' | 'herb' | 'drug'>('both')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (endpoint === 'api-info') return onSubmit({ endpoint })
    if (endpoint === 'safety-analyze') {
      const herbList = herbs.split(',').map((s) => s.trim()).filter(Boolean)
      const drugList = drugs.split(',').map((s) => s.trim()).filter(Boolean)
      if (!herbList.length || !drugList.length) return alert('Provide at least one herb and one drug')
      return onSubmit({ endpoint, herbs: herbList, drugs: drugList })
    }
    if (endpoint === 'search-interactions') {
      if (!query.trim()) return alert('Enter a search term')
      return onSubmit({ endpoint, search_term: query.trim(), search_type: searchType })
    }
  }

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <label>Endpoint:</label>
      <select value={endpoint} onChange={(e) => setEndpoint(e.target.value as Endpoint)}>
        <option value="api-info">API Info (GET /api-info)</option>
        <option value="safety-analyze">Safety Analysis (POST /safety-analysis/analyze)</option>
        <option value="search-interactions">Search Interactions (POST /safety-analysis/search-interactions)</option>
      </select>

      {endpoint === 'api-info' && (
        <div style={{marginLeft:8}}>No parameters — fetches API capabilities.</div>
      )}

      {endpoint === 'safety-analyze' && (
        <>
          <input placeholder="Herbs (comma-separated)" value={herbs} onChange={(e) => setHerbs(e.target.value)} />
          <input placeholder="Drugs (comma-separated)" value={drugs} onChange={(e) => setDrugs(e.target.value)} />
        </>
      )}

      {endpoint === 'search-interactions' && (
        <>
          <input placeholder="Search term (herb or drug)" value={query} onChange={(e) => setQuery(e.target.value)} />
          <select value={searchType} onChange={(e) => setSearchType(e.target.value as any)}>
            <option value="both">Both</option>
            <option value="herb">Herb</option>
            <option value="drug">Drug</option>
          </select>
        </>
      )}

      <button type="submit" disabled={loading}>{loading ? 'Loading…' : 'Run'}</button>
    </form>
  )
}
