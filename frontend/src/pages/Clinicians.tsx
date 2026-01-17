import React, { useState } from 'react'
import { useAuth, useAuthenticatedFetch } from '../contexts/AuthContext'

interface ClinicalQueryResult {
  concepts?: any[]
  cross_domain_mappings?: any[]
  safety_analysis?: any
  confidence_score?: number
  warnings?: string[]
  metadata?: any
}

interface PrescriptionAnalysis {
  request_id?: string
  entities?: Array<{
    entity_type: string
    text: string
    start_pos: number
    end_pos: number
    confidence: number
  }>
  confidence_score?: number
  processing_time?: number
  warnings?: string[]
  metadata?: {
    model_version?: string
    timestamp?: string
    input_length?: number
  }
  semantic_mappings?: Array<{
    allopathic_drug: string
    ayurvedic_alternatives: Array<{
      herb_name: string
      dosage: string
      mechanism: string
      confidence: number
    }>
  }>
  safety_assessment?: {
    overall_risk: string
    interactions_detected: any[]
    contraindications: string[]
    monitoring_requirements: string[]
  }
  recommendations?: Array<{
    type: string
    recommendation: string
    priority: string
    evidence_level: string
  }>
  analysis_summary?: {
    entities_found: number
    mappings_available: number
    safety_concerns: number
    recommendations_provided: number
  }
  // Legacy fields for backward compatibility
  parsed_prescription?: any
}

export default function Clinicians() {
  const { user } = useAuth()
  const authenticatedFetch = useAuthenticatedFetch()
  const [activeTab, setActiveTab] = useState<'query' | 'prescription' | 'safety'>('query')
  const [query, setQuery] = useState('')
  const [prescriptionText, setPrescriptionText] = useState('')
  const [herbs, setHerbs] = useState('')
  const [drugs, setDrugs] = useState('')
  const [results, setResults] = useState<ClinicalQueryResult | null>(null)
  const [prescriptionResults, setPrescriptionResults] = useState<PrescriptionAnalysis | null>(null)
  const [safetyResults, setSafetyResults] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleQuerySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await authenticatedFetch(`${baseUrl}/api/knowledge/query`, {
        method: 'POST',
        body: JSON.stringify({
          query_text: query,
          query_type: 'clinical',
          user_role: 'practitioner'
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

  const handlePrescriptionSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!prescriptionText.trim()) return

    setLoading(true)
    setError(null)

    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await authenticatedFetch(`${baseUrl}/api/prescription/parse`, {
        method: 'POST',
        body: JSON.stringify({
          text: prescriptionText,  // Changed from prescription_text to text
          metadata: {},
          validate_ontologies: true,
          enhance_confidence: true
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setPrescriptionResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleSafetySubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const herbList = herbs.split(',').map(s => s.trim()).filter(Boolean)
    const drugList = drugs.split(',').map(s => s.trim()).filter(Boolean)
    
    if (!herbList.length || !drugList.length) {
      setError('Please provide at least one herb and one drug')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await authenticatedFetch(`${baseUrl}/api/safety-analysis/analyze`, {
        method: 'POST',
        body: JSON.stringify({
          herbs: herbList,
          drugs: drugList
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setSafetyResults(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="clinicians-page">
      <div className="header">
        <h1>Ayurvedic Clinical Bridge - Practitioners Portal</h1>
        
        {user?.credentials && (
          <div className="practitioner-info">
            <div className="credentials-status">
              <h3>Practitioner Information</h3>
              <div className="credential-details">
                <p><strong>License:</strong> {user.credentials.license_number}</p>
                <p><strong>Specialization:</strong> {user.credentials.specialization}</p>
                <p><strong>Issuing Authority:</strong> {user.credentials.issuing_authority}</p>
                <p><strong>Status:</strong> 
                  <span className={`status-badge ${user.credentials.verification_status ? 'verified' : 'pending'}`}>
                    {user.credentials.verification_status ? 'Verified' : 'Pending Verification'}
                  </span>
                </p>
              </div>
            </div>
          </div>
        )}
        
        <div className="professional-notice">
          <p>
            <strong>For Qualified Healthcare Practitioners Only</strong><br/>
            This portal provides detailed clinical information for licensed healthcare providers.
            All recommendations should be evaluated within the context of individual patient care.
          </p>
        </div>
      </div>

      <div className="tabs">
        <button 
          className={activeTab === 'query' ? 'active' : ''}
          onClick={() => setActiveTab('query')}
        >
          Knowledge Query
        </button>
        <button 
          className={activeTab === 'prescription' ? 'active' : ''}
          onClick={() => setActiveTab('prescription')}
        >
          Prescription Analysis
        </button>
        <button 
          className={activeTab === 'safety' ? 'active' : ''}
          onClick={() => setActiveTab('safety')}
        >
          Safety Analysis
        </button>
      </div>

      {error && (
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}

      {activeTab === 'query' && (
        <div className="query-tab">
          <h2>Clinical Knowledge Query</h2>
          <form onSubmit={handleQuerySubmit} className="query-form">
            <div className="form-group">
              <label htmlFor="clinical-query">Clinical Query:</label>
              <textarea
                id="clinical-query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., Ayurvedic treatment protocols for diabetes, herb-drug interactions with metformin, dosage guidelines for Ashwagandha"
                rows={3}
                disabled={loading}
              />
            </div>
            <button type="submit" disabled={loading || !query.trim()}>
              {loading ? 'Analyzing...' : 'Query Knowledge Base'}
            </button>
          </form>

          {results && (
            <div className="clinical-results">
              <h3>Clinical Information</h3>
              
              {results.concepts && results.concepts.length > 0 && (
                <div className="clinical-concepts">
                  <h4>Relevant Clinical Concepts</h4>
                  {results.concepts.map((concept, index) => (
                    <div key={index} className="clinical-concept-card">
                      <h5>{concept.concept_name}</h5>
                      <p><strong>Category:</strong> {concept.concept_type}</p>
                      {concept.descriptions && (
                        <div>
                          <strong>Clinical Description:</strong>
                          {concept.descriptions.map((desc: string, i: number) => (
                            <p key={i}>{desc}</p>
                          ))}
                        </div>
                      )}
                      {concept.properties && Object.keys(concept.properties).length > 0 && (
                        <div>
                          <strong>Clinical Properties:</strong>
                          <ul>
                            {Object.entries(concept.properties).slice(0, 5).map(([key, value]) => (
                              <li key={key}><strong>{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:</strong> {String(value)}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {concept.relationships && concept.relationships.length > 0 && (
                        <div>
                          <strong>Clinical Relationships:</strong>
                          <ul>
                            {concept.relationships.slice(0, 3).map((rel: any, i: number) => (
                              <li key={i}>{rel.type.replace(/_/g, ' ')}: {rel.target_concept}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      <p><strong>Evidence Level:</strong> {concept.confidence_score ? (concept.confidence_score * 100).toFixed(1) + '%' : 'Not specified'}</p>
                      <p><strong>Sources:</strong> {concept.sources ? concept.sources.slice(0, 3).join(', ') : 'Classical Ayurvedic texts'}</p>
                    </div>
                  ))}
                </div>
              )}

              {results.cross_domain_mappings && results.cross_domain_mappings.length > 0 && (
                <div className="clinical-mappings">
                  <h4>Cross-Domain Clinical Mappings</h4>
                  {results.cross_domain_mappings.map((mapping, index) => (
                    <div key={index} className="clinical-mapping-card">
                      <p><strong>Biomedical Concept:</strong> {mapping.biomedical_concept}</p>
                      <p><strong>Ayurvedic Equivalent:</strong> {mapping.ayurvedic_concept}</p>
                      <p><strong>Mapping Confidence:</strong> {(mapping.confidence_score * 100).toFixed(1)}%</p>
                      {mapping.source_evidence && (
                        <p><strong>Evidence:</strong> {mapping.source_evidence.join('; ')}</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'prescription' && (
        <div className="prescription-tab">
          <h2>Prescription Analysis</h2>
          <form onSubmit={handlePrescriptionSubmit} className="prescription-form">
            <div className="form-group">
              <label htmlFor="prescription-text">Prescription Text:</label>
              <textarea
                id="prescription-text"
                value={prescriptionText}
                onChange={(e) => setPrescriptionText(e.target.value)}
                placeholder="Enter prescription text for analysis and Ayurvedic mapping..."
                rows={5}
                disabled={loading}
              />
            </div>
            <button type="submit" disabled={loading || !prescriptionText.trim()}>
              {loading ? 'Analyzing...' : 'Analyze Prescription'}
            </button>
          </form>

          {prescriptionResults && (
            <div className="prescription-results">
              <h3>Prescription Analysis Results</h3>
              
              {prescriptionResults.request_id && (
                <div className="analysis-info">
                  <p><strong>Request ID:</strong> {prescriptionResults.request_id}</p>
                  <p><strong>Processing Time:</strong> {prescriptionResults.processing_time ? prescriptionResults.processing_time.toFixed(3) + 's' : 'Not specified'}</p>
                  <p><strong>Overall Confidence:</strong> {prescriptionResults.confidence_score ? (prescriptionResults.confidence_score * 100).toFixed(1) + '%' : 'Not specified'}</p>
                </div>
              )}

              {prescriptionResults.entities && prescriptionResults.entities.length > 0 && (
                <div className="extracted-entities">
                  <h4>📋 Extracted Entities</h4>
                  <div className="entities-grid">
                    {prescriptionResults.entities.map((entity: any, index: number) => (
                      <div key={index} className="entity-card">
                        <div className="entity-header">
                          <span className={`entity-type ${entity.entity_type?.toLowerCase()}`}>
                            {entity.entity_type}
                          </span>
                          <span className="entity-confidence">
                            {(entity.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="entity-text">
                          <strong>Text:</strong> "{entity.text}"
                        </div>
                        <div className="entity-position">
                          <small>Position: {entity.start_pos}-{entity.end_pos}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {prescriptionResults.warnings && prescriptionResults.warnings.length > 0 && (
                <div className="analysis-warnings">
                  <h4>⚠️ Analysis Warnings</h4>
                  <ul>
                    {prescriptionResults.warnings.map((warning: string, index: number) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {prescriptionResults.metadata && (
                <div className="analysis-metadata">
                  <h4>Analysis Information</h4>
                  <div className="metadata-grid">
                    {prescriptionResults.metadata.model_version && (
                      <p><strong>Model Version:</strong> {prescriptionResults.metadata.model_version}</p>
                    )}
                    {prescriptionResults.metadata.timestamp && (
                      <p><strong>Timestamp:</strong> {prescriptionResults.metadata.timestamp}</p>
                    )}
                    {prescriptionResults.metadata.input_length && (
                      <p><strong>Input Length:</strong> {prescriptionResults.metadata.input_length} characters</p>
                    )}
                  </div>
                </div>
              )}

              {prescriptionResults.semantic_mappings && prescriptionResults.semantic_mappings.length > 0 && (
                <div className="semantic-mappings">
                  <h4>🌿 Ayurvedic Alternatives</h4>
                  {prescriptionResults.semantic_mappings.map((mapping: any, index: number) => (
                    <div key={index} className="mapping-card">
                      <div className="mapping-header">
                        <h5>Allopathic Drug: {mapping.allopathic_drug}</h5>
                      </div>
                      <div className="alternatives-list">
                        <strong>Ayurvedic Alternatives:</strong>
                        {mapping.ayurvedic_alternatives.map((alt: any, altIndex: number) => (
                          <div key={altIndex} className="alternative-item">
                            <div className="alt-header">
                              <span className="herb-name">{alt.herb_name}</span>
                              <span className="alt-confidence">{(alt.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <p><strong>Dosage:</strong> {alt.dosage}</p>
                            <p><strong>Mechanism:</strong> {alt.mechanism}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {prescriptionResults.safety_assessment && (
                <div className="safety-assessment">
                  <h4>🛡️ Safety Assessment</h4>
                  <div className="safety-details">
                    <div className={`risk-level ${prescriptionResults.safety_assessment.overall_risk}`}>
                      <strong>Overall Risk Level:</strong> {prescriptionResults.safety_assessment.overall_risk.toUpperCase()}
                    </div>
                    
                    {prescriptionResults.safety_assessment.contraindications && prescriptionResults.safety_assessment.contraindications.length > 0 && (
                      <div className="contraindications">
                        <strong>⚠️ Contraindications:</strong>
                        <ul>
                          {prescriptionResults.safety_assessment.contraindications.map((contra: string, index: number) => (
                            <li key={index}>{contra}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {prescriptionResults.safety_assessment.monitoring_requirements && prescriptionResults.safety_assessment.monitoring_requirements.length > 0 && (
                      <div className="monitoring-requirements">
                        <strong>📊 Monitoring Requirements:</strong>
                        <ul>
                          {prescriptionResults.safety_assessment.monitoring_requirements.map((req: string, index: number) => (
                            <li key={index}>{req}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {prescriptionResults.recommendations && prescriptionResults.recommendations.length > 0 && (
                <div className="prescription-recommendations">
                  <h4>💡 Clinical Recommendations</h4>
                  <div className="recommendations-grid">
                    {prescriptionResults.recommendations.map((rec: any, index: number) => (
                      <div key={index} className="recommendation-card">
                        <div className="rec-header">
                          <span className={`rec-type ${rec.type}`}>{rec.type.toUpperCase()}</span>
                          <span className={`priority-badge ${rec.priority}`}>{rec.priority}</span>
                        </div>
                        <p className="rec-text">{rec.recommendation}</p>
                        <div className="evidence-level">
                          <small>Evidence Level: {rec.evidence_level}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {prescriptionResults.analysis_summary && (
                <div className="analysis-summary">
                  <h4>📈 Analysis Summary</h4>
                  <div className="summary-stats">
                    <div className="stat-item">
                      <span className="stat-number">{prescriptionResults.analysis_summary.entities_found}</span>
                      <span className="stat-label">Entities Found</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{prescriptionResults.analysis_summary.mappings_available}</span>
                      <span className="stat-label">Ayurvedic Mappings</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{prescriptionResults.analysis_summary.safety_concerns}</span>
                      <span className="stat-label">Safety Concerns</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-number">{prescriptionResults.analysis_summary.recommendations_provided}</span>
                      <span className="stat-label">Recommendations</span>
                    </div>
                  </div>
                </div>
              )}

              <div className="clinical-disclaimer">
                <h4>🔒 Clinical Disclaimer</h4>
                <p>
                  This analysis is for clinical reference only. Always verify extracted information 
                  against the original prescription and consider individual patient factors when 
                  making treatment decisions.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'safety' && (
        <div className="safety-tab">
          <h2>Herb-Drug Interaction Analysis</h2>
          <form onSubmit={handleSafetySubmit} className="safety-form">
            <div className="form-group">
              <label htmlFor="herbs-input">Herbs (comma-separated):</label>
              <input
                id="herbs-input"
                type="text"
                value={herbs}
                onChange={(e) => setHerbs(e.target.value)}
                placeholder="e.g., Ashwagandha, Turmeric, Ginkgo"
                disabled={loading}
              />
            </div>
            <div className="form-group">
              <label htmlFor="drugs-input">Drugs (comma-separated):</label>
              <input
                id="drugs-input"
                type="text"
                value={drugs}
                onChange={(e) => setDrugs(e.target.value)}
                placeholder="e.g., Warfarin, Metformin, Lisinopril"
                disabled={loading}
              />
            </div>
            <button type="submit" disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze Interactions'}
            </button>
          </form>

          {safetyResults && (
            <div className="safety-results">
              <h3>Safety Analysis Results</h3>
              
              {safetyResults.request_id && (
                <div className="analysis-info">
                  <p><strong>Analysis ID:</strong> {safetyResults.request_id}</p>
                </div>
              )}

              {safetyResults.interactions && safetyResults.interactions.length > 0 && (
                <div className="interactions-section">
                  <h4>🚨 Potential Interactions Detected</h4>
                  {safetyResults.interactions.map((interaction: any, index: number) => (
                    <div key={index} className="interaction-card">
                      <div className="interaction-header">
                        <h5>{interaction.herb} ↔ {interaction.drug}</h5>
                        <span className={`severity-badge ${interaction.severity?.toLowerCase()}`}>
                          {interaction.severity || 'Unknown'}
                        </span>
                      </div>
                      
                      <div className="interaction-details">
                        <p><strong>Interaction Type:</strong> {interaction.interaction_type}</p>
                        <p><strong>Confidence:</strong> {interaction.confidence ? (interaction.confidence * 100).toFixed(1) + '%' : 'Not specified'}</p>
                        
                        {interaction.description && (
                          <div className="interaction-description">
                            <strong>Description:</strong>
                            <p>{interaction.description}</p>
                          </div>
                        )}
                        
                        {interaction.recommendation && (
                          <div className="interaction-recommendation">
                            <strong>Recommendation:</strong>
                            <p>{interaction.recommendation}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {safetyResults.overall_risk_level && (
                <div className="risk-assessment">
                  <h4>Overall Risk Assessment</h4>
                  <div className={`risk-level ${safetyResults.overall_risk_level.toLowerCase()}`}>
                    <strong>Risk Level:</strong> {safetyResults.overall_risk_level}
                  </div>
                  <p><strong>Confidence Score:</strong> {safetyResults.confidence_score ? (safetyResults.confidence_score * 100).toFixed(1) + '%' : 'Not specified'}</p>
                  <p><strong>Processing Time:</strong> {safetyResults.processing_time ? safetyResults.processing_time.toFixed(3) + 's' : 'Not specified'}</p>
                </div>
              )}

              {safetyResults.warnings && safetyResults.warnings.length > 0 && (
                <div className="analysis-warnings">
                  <h4>⚠️ Important Warnings</h4>
                  <ul>
                    {safetyResults.warnings.map((warning: string, index: number) => (
                      <li key={index}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {safetyResults.recommendations && safetyResults.recommendations.length > 0 && (
                <div className="general-recommendations">
                  <h4>📋 General Recommendations</h4>
                  <ul>
                    {safetyResults.recommendations.map((rec: string, index: number) => (
                      <li key={index}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}

              {safetyResults.metadata && (
                <div className="analysis-metadata">
                  <h4>Analysis Information</h4>
                  <p><strong>Analysis Version:</strong> {safetyResults.metadata.analysis_version || 'Not specified'}</p>
                  {safetyResults.metadata.data_sources && (
                    <p><strong>Data Sources:</strong> {safetyResults.metadata.data_sources.join(', ')}</p>
                  )}
                </div>
              )}

              <div className="clinical-disclaimer">
                <h4>🔒 Clinical Disclaimer</h4>
                <p>
                  This analysis is for clinical reference only. Always consider individual patient factors, 
                  medical history, and current clinical guidelines when making treatment decisions. 
                  Consult additional resources and specialist colleagues when needed.
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}