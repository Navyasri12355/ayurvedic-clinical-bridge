/**
 * Intelligent Results Component
 *
 * Displays results from the intelligent routing system.
 */

import React, { useState } from 'react'
import { IntelligentQueryResponse } from '../services/intelligentQueryService'
import ExplanationViewer from './ExplanationViewer'

interface Props {
  results: IntelligentQueryResponse | any
  showRoutingInfo?: boolean
}

export default function IntelligentResults({ results, showRoutingInfo = false }: Props) {
  const [activeTab, setActiveTab] = useState<
    'overview' | 'entities' | 'knowledge' | 'interactions' | 'safety' | 'treatments' | 'routing' | 'explanation'
  >('overview')

  const isConversational = results.conversational === true

  // -----------------------------------------------------------------------
  // Conversational (new format)
  // -----------------------------------------------------------------------
  if (isConversational) {
    return (
      <div className="conversational-results">
        <div className="response-header">
          <h3>🤖 AI Response</h3>
          <div className="response-meta">
            <span>Query Type: {results.query_type?.replace('_', ' ') || 'General'}</span>
            {results.processing_time && (
              <span>Processing Time: {results.processing_time.toFixed(3)}s</span>
            )}
          </div>
        </div>

        <div className="response-content">
          <div className="response-text">
            {(results.response as string).split('\n').map((paragraph: string, index: number) => {
              if (!paragraph.trim()) return <br key={index} />
              const html = paragraph.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              if (paragraph.trim().startsWith('•')) {
                return (
                  <div key={index} className="bullet-point">
                    <span dangerouslySetInnerHTML={{ __html: html }} />
                  </div>
                )
              }
              return <p key={index} dangerouslySetInnerHTML={{ __html: html }} />
            })}
          </div>
        </div>

        {results.top_prediction && (
          <div className="prediction-summary">
            <h4>📊 Quick Summary</h4>
            <div className="prediction-info">
              <span className="prediction-label">Top Prediction:</span>
              <span className="prediction-value">{results.top_prediction}</span>
              {results.confidence && (
                <span className="prediction-confidence">({results.confidence})</span>
              )}
            </div>
          </div>
        )}

        {results.explanation && (
          <ExplanationViewer
            explanation={results.explanation}
            prediction={
              results.top_prediction
                ? { disease: results.top_prediction, confidence: results.confidence || 'Unknown' }
                : undefined
            }
          />
        )}

        <style>{`
          .conversational-results { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); overflow: hidden; margin-top: 20px; }
          .response-header { background: #f8f9fa; padding: 20px; border-bottom: 1px solid #e0e0e0; }
          .response-header h3 { margin: 0 0 10px 0; color: #333; font-size: 1.4em; }
          .response-meta { display: flex; gap: 20px; font-size: .9em; color: #666; }
          .response-content { padding: 25px; }
          .response-text { line-height: 1.6; color: #333; font-size: 1.05em; }
          .response-text p { margin: 0 0 15px 0; }
          .response-text strong { color: #007bff; font-weight: 600; }
          .bullet-point { margin: 8px 0 8px 20px; position: relative; }
          .bullet-point::before { content: "•"; color: #007bff; font-weight: bold; position: absolute; left: -15px; }
          .prediction-summary { background: #f0f8ff; border-top: 1px solid #e0e0e0; padding: 20px; }
          .prediction-summary h4 { margin: 0 0 15px 0; color: #333; }
          .prediction-info { display: flex; align-items: center; gap: 10px; }
          .prediction-label { font-weight: 600; color: #555; }
          .prediction-value { background: #007bff; color: white; padding: 4px 12px; border-radius: 15px; font-weight: 500; }
          .prediction-confidence { color: #666; font-size: .9em; }
        `}</style>
      </div>
    )
  }

  // -----------------------------------------------------------------------
  // Legacy complex format
  // -----------------------------------------------------------------------

  const getModelIcon = (model: string) => {
    if (model === 'bilstm_crf') return '⚡'
    if (model === 'biobert') return '🔬'
    return '🤖'
  }

  const getModelDescription = (model: string) => {
    if (model === 'bilstm_crf') return 'BiLSTM-CRF (Fast Entity Recognition)'
    if (model === 'biobert') return 'BioBERT (High-Accuracy Clinical Analysis)'
    return model || 'Unknown Model'
  }

  const getRiskColor = (level?: string | null) => {
    switch ((level || '').toLowerCase()) {
      case 'low':      return '#4caf50'
      case 'moderate': return '#ff9800'
      case 'high':     return '#f44336'
      case 'severe':   return '#d32f2f'
      default:         return '#757575'
    }
  }

  const hasContent = (section: string) => {
    switch (section) {
      case 'entities':     return (results.entities?.length ?? 0) > 0
      case 'knowledge':    return (results.knowledge_results?.length ?? 0) > 0
      case 'interactions': return (results.interactions?.length ?? 0) > 0
      case 'safety':       return !!results.safety_assessment
      case 'treatments':   return (results.treatment_recommendations?.length ?? 0) > 0
      case 'routing':      return !!results.routing_decision
      default:             return true
    }
  }

  const tabCount = (section: string) => {
    switch (section) {
      case 'entities':     return results.entities?.length ?? 0
      case 'knowledge':    return results.knowledge_results?.length ?? 0
      case 'interactions': return results.interactions?.length ?? 0
      case 'treatments':   return results.treatment_recommendations?.length ?? 0
      default:             return 0
    }
  }

  return (
    <div className="intelligent-results">
      {/* Header */}
      <div className="results-header">
        <div className="model-info">
          <span className="model-icon">{getModelIcon(results.model_used)}</span>
          <div className="model-details">
            <h3>Query Results</h3>
            <p className="model-description">{getModelDescription(results.model_used)}</p>
            <div className="processing-info">
              <span>Processing: {(results.processing_time || 0).toFixed(3)}s</span>
              <span>Role: {(results.user_role || 'user').replace('_', ' ')}</span>
            </div>
          </div>
        </div>

        {results.confidence_scores && (
          <div className="confidence-scores">
            {Object.entries(results.confidence_scores).map(([key, score]) => (
              <div key={key} className="confidence-item">
                <span className="confidence-label">
                  {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </span>
                <div className="confidence-bar">
                  <div className="confidence-fill" style={{ width: `${(score as number) * 100}%` }} />
                  <span className="confidence-value">{((score as number) * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="results-tabs">
        {(['overview', 'entities', 'knowledge', 'interactions', 'safety', 'treatments'] as const).map(tab => {
          if (tab !== 'overview' && !hasContent(tab)) return null
          return (
            <button
              key={tab}
              className={`tab ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
              {tabCount(tab) > 0 && ` (${tabCount(tab)})`}
            </button>
          )
        })}
        {showRoutingInfo && hasContent('routing') && (
          <button className={`tab ${activeTab === 'routing' ? 'active' : ''}`} onClick={() => setActiveTab('routing')}>
            Routing Info
          </button>
        )}
        {results.explanation && (
          <button className={`tab ${activeTab === 'explanation' ? 'active' : ''}`} onClick={() => setActiveTab('explanation')}>
            AI Explanation
          </button>
        )}
      </div>

      {/* Content */}
      <div className="results-content">
        {activeTab === 'overview' && (
          <div className="overview-grid">
            <div className="overview-card">
              <h4>📊 Analysis Summary</h4>
              <div className="summary-stats">
                {[
                  ['entities', 'Entities Found'],
                  ['knowledge_results', 'Knowledge Results'],
                  ['interactions', 'Interactions'],
                  ['treatment_recommendations', 'Treatments'],
                ].map(([key, label]) => (
                  <div key={key} className="stat">
                    <span className="stat-number">{(results as any)[key]?.length ?? 0}</span>
                    <span className="stat-label">{label}</span>
                  </div>
                ))}
              </div>
            </div>
            {results.safety_assessment && (
              <div className="overview-card">
                <h4>🛡️ Safety Overview</h4>
                <div
                  className="risk-indicator"
                  style={{ backgroundColor: getRiskColor(results.safety_assessment.risk_level) }}
                >
                  {(results.safety_assessment.risk_level || 'UNKNOWN').toUpperCase()} RISK
                </div>
                <p>Risk Score: {((results.safety_assessment.overall_risk_score || 0) * 100).toFixed(1)}%</p>
                {results.safety_assessment.requires_consultation && (
                  <p className="consultation-required">⚠️ Professional consultation recommended</p>
                )}
              </div>
            )}
            <div className="overview-card">
              <h4>🎯 Query Analysis</h4>
              <p><strong>Query:</strong> {results.query}</p>
              <p><strong>Model:</strong> {getModelDescription(results.model_used)}</p>
              <p><strong>Time:</strong> {(results.processing_time || 0).toFixed(3)}s</p>
            </div>
          </div>
        )}

        {activeTab === 'entities' && (
          <div>
            <h4>🏷️ Extracted Entities</h4>
            <div className="entities-grid">
              {(results.entities || []).map((e: any, i: number) => (
                <div key={i} className="entity-card">
                  <div className="entity-header">
                    <span className="entity-type">{e.type || 'UNKNOWN'}</span>
                    <span className="entity-confidence">{((e.confidence || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="entity-text">"{e.text || ''}"</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'treatments' && (
          <div>
            <h4>💊 Treatment Recommendations</h4>
            <div className="treatments-grid">
              {(results.treatment_recommendations || []).map((t: any, i: number) => (
                <div key={i} className="treatment-card">
                  <div className="treatment-header">
                    <h5>{t.condition || 'General Treatment'}</h5>
                    <span className="treatment-confidence">{((t.confidence || 0) * 100).toFixed(1)}% confidence</span>
                  </div>
                  <div className="treatment-details">
                    {t.herb && <p><strong>🌿 Herb:</strong> {t.herb}{t.sanskrit_name && ` (${t.sanskrit_name})`}</p>}
                    {t.dosage && <p><strong>💊 Dosage:</strong> {t.dosage}</p>}
                    {t.mechanism && <p><strong>🔬 Mechanism:</strong> {t.mechanism}</p>}
                    {t.evidence_level && <p><strong>📚 Evidence:</strong> {t.evidence_level.replace('_', ' ')}</p>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'interactions' && (
          <div>
            <h4>⚠️ Herb-Drug Interactions</h4>
            <div className="interactions-grid">
              {(results.interactions || []).map((ia: any, i: number) => (
                <div key={i} className="interaction-card">
                  <div className="interaction-header">
                    <h5>{ia.herb} + {ia.drug}</h5>
                    <span className="severity-badge" style={{ backgroundColor: getRiskColor(ia.severity) }}>
                      {(ia.severity || 'UNKNOWN').toUpperCase()}
                    </span>
                  </div>
                  <p><strong>Mechanism:</strong> {ia.mechanism}</p>
                  <p><strong>Confidence:</strong> {((ia.confidence || 0) * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'safety' && results.safety_assessment && (
          <div>
            <h4>🛡️ Safety Assessment</h4>
            <div
              className="risk-level"
              style={{ backgroundColor: getRiskColor(results.safety_assessment.risk_level) }}
            >
              {(results.safety_assessment.risk_level || 'UNKNOWN').toUpperCase()} RISK
            </div>
            <p>Overall Risk: {((results.safety_assessment.overall_risk_score || 0) * 100).toFixed(1)}%</p>
            {results.safety_assessment.recommendations?.length > 0 && (
              <ul>{results.safety_assessment.recommendations.map((r: string, i: number) => <li key={i}>{r}</li>)}</ul>
            )}
          </div>
        )}

        {activeTab === 'routing' && results.routing_decision && (
          <div>
            <h3>🎯 Routing Decision</h3>
            <p><strong>Model:</strong> {getModelDescription(results.routing_decision.model_used)}</p>
            <p><strong>Confidence:</strong> {(results.routing_decision.confidence * 100).toFixed(1)}%</p>
            <p><strong>Reasoning:</strong> {results.routing_decision.reasoning}</p>
          </div>
        )}

        {activeTab === 'explanation' && results.explanation && (
          <ExplanationViewer explanation={results.explanation} />
        )}
      </div>

      <style>{`
        .intelligent-results { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,.1); overflow: hidden; }
        .results-header { background: #f8f9fa; padding: 20px; border-bottom: 1px solid #e0e0e0; }
        .model-info { display: flex; align-items: flex-start; gap: 15px; margin-bottom: 20px; }
        .model-icon { font-size: 2em; }
        .model-details h3 { margin: 0 0 5px 0; }
        .model-description { color: #666; margin: 0 0 10px 0; }
        .processing-info { display: flex; gap: 20px; font-size: .9em; color: #777; }
        .confidence-scores { display: flex; flex-direction: column; gap: 10px; }
        .confidence-item { display: flex; align-items: center; gap: 10px; }
        .confidence-label { min-width: 120px; font-size: .9em; color: #555; }
        .confidence-bar { flex: 1; height: 20px; background: #e0e0e0; border-radius: 10px; position: relative; overflow: hidden; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #4caf50, #8bc34a); }
        .confidence-value { position: absolute; right: 8px; top: 50%; transform: translateY(-50%); font-size: .8em; font-weight: bold; }
        .results-tabs { display: flex; background: #f0f0f0; border-bottom: 1px solid #e0e0e0; overflow-x: auto; }
        .tab { background: none; border: none; padding: 12px 20px; cursor: pointer; white-space: nowrap; color: #666; font-weight: 500; transition: all .2s; }
        .tab:hover { background: #e8e8e8; color: #333; }
        .tab.active { background: white; color: #007bff; border-bottom: 2px solid #007bff; }
        .results-content { padding: 20px; }
        .overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .overview-card { background: #f8f9fa; border-radius: 6px; padding: 15px; }
        .overview-card h4 { margin: 0 0 15px 0; }
        .summary-stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .stat { text-align: center; }
        .stat-number { display: block; font-size: 2em; font-weight: bold; color: #007bff; }
        .stat-label { font-size: .9em; color: #666; }
        .risk-indicator, .risk-level { color: white; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: .9em; display: inline-block; margin-bottom: 8px; }
        .consultation-required { color: #856404; font-weight: 500; }
        .entities-grid, .treatments-grid, .interactions-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; }
        .entity-card, .treatment-card, .interaction-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; }
        .entity-header, .interaction-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .entity-type { background: #e3f2fd; color: #1976d2; padding: 4px 8px; border-radius: 12px; font-size: .8em; font-weight: 500; }
        .entity-confidence, .treatment-confidence { font-size: .9em; color: #666; }
        .entity-text, .treatment-header h5 { font-weight: 500; margin: 0 0 5px 0; }
        .severity-badge { color: white; padding: 4px 8px; border-radius: 12px; font-size: .8em; font-weight: 500; }
      `}</style>
    </div>
  )
}