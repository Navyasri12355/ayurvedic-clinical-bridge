/**
 * Intelligent Results Component
 * 
 * This component displays results from the intelligent routing system,
 * showing different types of information based on which model was used
 * and what type of analysis was performed.
 */

import React, { useState } from 'react'
import { IntelligentQueryResponse } from '../services/intelligentQueryService'
import ExplanationViewer from './ExplanationViewer'

interface Props {
  results: IntelligentQueryResponse | any // Allow both old and new formats
  showRoutingInfo?: boolean
}

export default function IntelligentResults({ results, showRoutingInfo = false }: Props) {
  const [activeTab, setActiveTab] = useState<'overview' | 'entities' | 'knowledge' | 'interactions' | 'safety' | 'treatments' | 'routing' | 'explanation'>('overview')

  // Check if this is the new conversational format
  const isConversational = results.conversational === true

  if (isConversational) {
    // Render conversational response
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
            {results.response.split('\n').map((paragraph: string, index: number) => {
              if (paragraph.trim() === '') return <br key={index} />
              
              // Handle markdown-style formatting
              let formattedText = paragraph
              
              // Bold text
              formattedText = formattedText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
              
              // Bullet points
              if (paragraph.trim().startsWith('•')) {
                return (
                  <div key={index} className="bullet-point">
                    <span dangerouslySetInnerHTML={{ __html: formattedText }} />
                  </div>
                )
              }
              
              return (
                <p key={index} dangerouslySetInnerHTML={{ __html: formattedText }} />
              )
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
            prediction={results.top_prediction ? {
              disease: results.top_prediction,
              confidence: results.confidence || 'Unknown'
            } : undefined}
            inputText={results.query || 'Unknown input'}
            showTechnicalDetails={true}
          />
        )}

        <style>{`
          .conversational-results {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-top: 20px;
          }

          .response-header {
            background: #f8f9fa;
            padding: 20px;
            border-bottom: 1px solid #e0e0e0;
          }

          .response-header h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.4em;
          }

          .response-meta {
            display: flex;
            gap: 20px;
            font-size: 0.9em;
            color: #666;
          }

          .response-content {
            padding: 25px;
          }

          .response-text {
            line-height: 1.6;
            color: #333;
            font-size: 1.05em;
          }

          .response-text p {
            margin: 0 0 15px 0;
          }

          .response-text strong {
            color: #007bff;
            font-weight: 600;
          }

          .bullet-point {
            margin: 8px 0 8px 20px;
            position: relative;
          }

          .bullet-point::before {
            content: "•";
            color: #007bff;
            font-weight: bold;
            position: absolute;
            left: -15px;
          }

          .prediction-summary {
            background: #f0f8ff;
            border-top: 1px solid #e0e0e0;
            padding: 20px;
          }

          .prediction-summary h4 {
            margin: 0 0 15px 0;
            color: #333;
          }

          .prediction-info {
            display: flex;
            align-items: center;
            gap: 10px;
          }

          .prediction-label {
            font-weight: 600;
            color: #555;
          }

          .prediction-value {
            background: #007bff;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-weight: 500;
          }

          .prediction-confidence {
            color: #666;
            font-size: 0.9em;
          }
        `}</style>
      </div>
    )
  }

  // Original complex format handling (for backward compatibility)
  const getModelIcon = (model: string) => {
    switch (model) {
      case 'knowledge_base': return '🧠'
      case 'bilstm_crf': return '⚡'
      case 'biobert': return '🔬'
      default: return '🤖'
    }
  }

  const getModelDescription = (model: string) => {
    switch (model) {
      case 'bilstm_crf': return 'BiLSTM-CRF (Fast Entity Recognition)'
      case 'biobert': return 'BioBERT (High-Accuracy Clinical Analysis)'
      default: return 'Unknown Model'
    }
  }

  const getRiskLevelColor = (level: string | undefined | null) => {
    if (!level) return '#757575' // Default gray color for undefined/null
    
    switch (level.toLowerCase()) {
      case 'low': return '#4caf50'
      case 'moderate': return '#ff9800'
      case 'high': return '#f44336'
      case 'severe': return '#d32f2f'
      default: return '#757575'
    }
  }

  const getEvidenceLevelBadge = (level: string) => {
    const colors = {
      'traditional': '#8bc34a',
      'clinical_study': '#2196f3',
      'systematic_review': '#9c27b0',
      'meta_analysis': '#e91e63',
      'expert_consensus': '#ff5722'
    }
    return colors[level as keyof typeof colors] || '#757575'
  }

  const hasContent = (section: string) => {
    switch (section) {
      case 'entities': return results.entities?.length > 0
      case 'knowledge': return results.knowledge_results?.length > 0
      case 'interactions': return results.interactions?.length > 0
      case 'safety': return !!results.safety_assessment
      case 'treatments': return results.treatment_recommendations?.length > 0
      case 'routing': return !!results.routing_decision
      default: return true
    }
  }

  const getTabCount = (section: string) => {
    switch (section) {
      case 'entities': return results.entities?.length || 0
      case 'knowledge': return results.knowledge_results?.length || 0
      case 'interactions': return results.interactions?.length || 0
      case 'treatments': return results.treatment_recommendations?.length || 0
      default: return 0
    }
  }

  return (
    <div className="intelligent-results">
      <div className="results-header">
        <div className="model-info">
          <span className="model-icon">{getModelIcon(results.model_used || 'unknown')}</span>
          <div className="model-details">
            <h3>Query Results</h3>
            <p className="model-description">
              {getModelDescription(results.model_used || 'unknown')}
            </p>
            <div className="processing-info">
              <span>Processing Time: {(results.processing_time || 0).toFixed(3)}s</span>
              <span>User Role: {(results.user_role || 'user').replace('_', ' ')}</span>
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
                  <div 
                    className="confidence-fill" 
                    style={{ width: `${(score as number) * 100}%` }}
                  ></div>
                  <span className="confidence-value">{((score as number) * 100).toFixed(1)}%</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="results-tabs">
        <button 
          className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
          onClick={() => setActiveTab('overview')}
        >
          Overview
        </button>
        
        {hasContent('entities') && (
          <button 
            className={`tab ${activeTab === 'entities' ? 'active' : ''}`}
            onClick={() => setActiveTab('entities')}
          >
            Entities ({getTabCount('entities')})
          </button>
        )}
        
        {hasContent('knowledge') && (
          <button 
            className={`tab ${activeTab === 'knowledge' ? 'active' : ''}`}
            onClick={() => setActiveTab('knowledge')}
          >
            Knowledge ({getTabCount('knowledge')})
          </button>
        )}
        
        {hasContent('interactions') && (
          <button 
            className={`tab ${activeTab === 'interactions' ? 'active' : ''}`}
            onClick={() => setActiveTab('interactions')}
          >
            Interactions ({getTabCount('interactions')})
          </button>
        )}
        
        {hasContent('safety') && (
          <button 
            className={`tab ${activeTab === 'safety' ? 'active' : ''}`}
            onClick={() => setActiveTab('safety')}
          >
            Safety Assessment
          </button>
        )}
        
        {hasContent('treatments') && (
          <button 
            className={`tab ${activeTab === 'treatments' ? 'active' : ''}`}
            onClick={() => setActiveTab('treatments')}
          >
            Treatments ({getTabCount('treatments')})
          </button>
        )}
        
        {showRoutingInfo && hasContent('routing') && (
          <button 
            className={`tab ${activeTab === 'routing' ? 'active' : ''}`}
            onClick={() => setActiveTab('routing')}
          >
            Routing Info
          </button>
        )}
        
        {results.explanation && (
          <button 
            className={`tab ${activeTab === 'explanation' ? 'active' : ''}`}
            onClick={() => setActiveTab('explanation')}
          >
            AI Explanation
          </button>
        )}
      </div>

      <div className="results-content">
        {activeTab === 'overview' && (
          <div className="overview-tab">
            <div className="overview-grid">
              <div className="overview-card">
                <h4>📊 Analysis Summary</h4>
                <div className="summary-stats">
                  <div className="stat">
                    <span className="stat-number">{results.entities?.length || 0}</span>
                    <span className="stat-label">Entities Found</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">{results.knowledge_results?.length || 0}</span>
                    <span className="stat-label">Knowledge Results</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">{results.interactions?.length || 0}</span>
                    <span className="stat-label">Interactions</span>
                  </div>
                  <div className="stat">
                    <span className="stat-number">{results.treatment_recommendations?.length || 0}</span>
                    <span className="stat-label">Treatments</span>
                  </div>
                </div>
              </div>

              {results.safety_assessment && (
                <div className="overview-card">
                  <h4>🛡️ Safety Overview</h4>
                  <div className="safety-overview">
                    <div 
                      className="risk-indicator"
                      style={{ backgroundColor: getRiskLevelColor(results.safety_assessment.risk_level) }}
                    >
                      {(results.safety_assessment.risk_level || 'UNKNOWN').toUpperCase()} RISK
                    </div>
                    <p>Risk Score: {((results.safety_assessment.overall_risk_score || 0) * 100).toFixed(1)}%</p>
                    {results.safety_assessment.requires_consultation && (
                      <p className="consultation-required">⚠️ Professional consultation recommended</p>
                    )}
                  </div>
                </div>
              )}

              <div className="overview-card">
                <h4>🎯 Query Analysis</h4>
                <div className="query-analysis">
                  <p><strong>Original Query:</strong> {results.query}</p>
                  <p><strong>Model Used:</strong> {getModelDescription(results.model_used || 'unknown')}</p>
                  <p><strong>Processing Time:</strong> {(results.processing_time || 0).toFixed(3)}s</p>
                  {results.routing_decision && (
                    <p><strong>Routing Reason:</strong> {results.routing_decision.reasoning}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Rest of the tabs remain the same but with safe property access */}
        {activeTab === 'entities' && results.entities?.length > 0 && (
          <div className="entities-tab">
            <h4>🏷️ Extracted Entities</h4>
            <div className="entities-grid">
              {results.entities.map((entity: any, index: number) => (
                <div key={index} className="entity-card">
                  <div className="entity-header">
                    <span className={`entity-type ${(entity.type || '').toLowerCase()}`}>
                      {entity.type || 'UNKNOWN'}
                    </span>
                    <span className="entity-confidence">
                      {((entity.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="entity-text">"{entity.text || ''}"</div>
                  {entity.start !== undefined && entity.end !== undefined && (
                    <div className="entity-position">
                      Position: {entity.start}-{entity.end}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'treatments' && results.treatment_recommendations?.length > 0 && (
          <div className="treatments-tab">
            <h4>💊 Treatment Recommendations</h4>
            <div className="treatments-grid">
              {results.treatment_recommendations.map((treatment: any, index: number) => (
                <div key={index} className="treatment-card">
                  <div className="treatment-header">
                    <h5 className="treatment-condition">{treatment.condition || 'General Treatment'}</h5>
                    <span className="treatment-confidence">
                      {((treatment.confidence || 0) * 100).toFixed(1)}% confidence
                    </span>
                  </div>
                  
                  <div className="treatment-details">
                    <div className="treatment-herb">
                      <strong>🌿 Herb/Treatment:</strong>
                      <span>{treatment.herb || 'Not specified'}</span>
                      {treatment.sanskrit_name && (
                        <small className="sanskrit-name">({treatment.sanskrit_name})</small>
                      )}
                    </div>
                    
                    {treatment.dosage && (
                      <div className="treatment-dosage">
                        <strong>💊 Dosage:</strong>
                        <span>{treatment.dosage}</span>
                      </div>
                    )}
                    
                    {treatment.formulation && (
                      <div className="treatment-formulation">
                        <strong>🧪 Formulation:</strong>
                        <span>{treatment.formulation}</span>
                      </div>
                    )}
                    
                    {treatment.duration && (
                      <div className="treatment-duration">
                        <strong>⏱️ Duration:</strong>
                        <span>{treatment.duration}</span>
                      </div>
                    )}
                    
                    {treatment.mechanism && (
                      <div className="treatment-mechanism">
                        <strong>🔬 Mechanism:</strong>
                        <span>{treatment.mechanism}</span>
                      </div>
                    )}
                    
                    {treatment.clinical_evidence && (
                      <div className="treatment-clinical-evidence">
                        <strong>📊 Clinical Evidence:</strong>
                        <span>{treatment.clinical_evidence}</span>
                      </div>
                    )}
                    
                    {treatment.evidence_level && (
                      <div className="treatment-evidence">
                        <strong>📚 Evidence Level:</strong>
                        <span 
                          className="evidence-badge"
                          style={{ backgroundColor: getEvidenceLevelBadge(treatment.evidence_level) }}
                        >
                          {treatment.evidence_level.replace('_', ' ').toUpperCase()}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'knowledge' && results.knowledge_results?.length > 0 && (
          <div className="knowledge-tab">
            <h4>📚 Knowledge Results</h4>
            <div className="knowledge-grid">
              {results.knowledge_results.map((knowledge: any, index: number) => (
                <div key={index} className="knowledge-card">
                  <div className="knowledge-header">
                    <h5>{knowledge.title || 'Knowledge Item'}</h5>
                    <span className="knowledge-confidence">
                      {((knowledge.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="knowledge-content">
                    {typeof knowledge.content === 'string' ? (
                      <p>{knowledge.content}</p>
                    ) : (
                      <div>
                        {Object.entries(knowledge.content || {}).map(([key, value]) => (
                          <p key={key}><strong>{key}:</strong> {String(value)}</p>
                        ))}
                      </div>
                    )}
                  </div>
                  {knowledge.source && (
                    <div className="knowledge-source">
                      <strong>Source:</strong> {knowledge.source}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'interactions' && results.interactions?.length > 0 && (
          <div className="interactions-tab">
            <h4>⚠️ Herb-Drug Interactions</h4>
            <div className="interactions-grid">
              {results.interactions.map((interaction: any, index: number) => (
                <div key={index} className="interaction-card">
                  <div className="interaction-header">
                    <h5>{interaction.herb} + {interaction.drug}</h5>
                    <span 
                      className="severity-badge"
                      style={{ backgroundColor: getRiskLevelColor(interaction.severity) }}
                    >
                      {(interaction.severity || 'UNKNOWN').toUpperCase()}
                    </span>
                  </div>
                  <div className="interaction-details">
                    <p><strong>Mechanism:</strong> {interaction.mechanism}</p>
                    <p><strong>Confidence:</strong> {((interaction.confidence || 0) * 100).toFixed(1)}%</p>
                    {interaction.evidence_level && (
                      <p><strong>Evidence:</strong> {interaction.evidence_level.replace('_', ' ')}</p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'safety' && results.safety_assessment && (
          <div className="safety-tab">
            <h4>🛡️ Safety Assessment</h4>
            <div className="safety-content">
              <div className="safety-overview">
                <div className="risk-summary">
                  <div 
                    className="risk-level"
                    style={{ backgroundColor: getRiskLevelColor(results.safety_assessment.risk_level) }}
                  >
                    {(results.safety_assessment.risk_level || 'UNKNOWN').toUpperCase()} RISK
                  </div>
                  <div className="risk-score">
                    Overall Risk Score: {((results.safety_assessment.overall_risk_score || 0) * 100).toFixed(1)}%
                  </div>
                </div>
                
                {results.safety_assessment.risk_factors && (
                  <div className="risk-factors">
                    <h5>Risk Factors:</h5>
                    {Object.entries(results.safety_assessment.risk_factors).map(([factor, score]) => (
                      <div key={factor} className="risk-factor">
                        <span>{factor.replace('_', ' ')}</span>
                        <span>{((score as number) * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
                
                {results.safety_assessment.recommendations && (
                  <div className="safety-recommendations">
                    <h5>Recommendations:</h5>
                    <ul>
                      {results.safety_assessment.recommendations.map((rec: string, index: number) => (
                        <li key={index}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
                
                {results.safety_assessment.requires_consultation && (
                  <div className="consultation-warning">
                    ⚠️ Professional consultation is recommended
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'routing' && results.routing_decision && (
          <div className="routing-tab">
            <div className="routing-info">
              <h3>🎯 Routing Decision</h3>
              <div className="routing-details">
                <div className="routing-card">
                  <h4>Model Selection</h4>
                  <p><strong>Selected Model:</strong> {getModelDescription(results.routing_decision.model_used)}</p>
                  <p><strong>Confidence:</strong> {(results.routing_decision.confidence * 100).toFixed(1)}%</p>
                  <p><strong>Reasoning:</strong> {results.routing_decision.reasoning}</p>
                </div>
                
                <div className="routing-card">
                  <h4>Query Analysis</h4>
                  <p><strong>Complexity:</strong> {results.routing_decision.query_complexity}</p>
                  <p><strong>Category:</strong> {results.routing_decision.query_category}</p>
                  <p><strong>Estimated Time:</strong> {results.routing_decision.estimated_processing_time.toFixed(3)}s</p>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'explanation' && results.explanation && (
          <div className="explanation-tab">
            <div className="explanation-content">
              <h3>🧠 AI Model Explanation</h3>
              
              <div className="explanation-section">
                <h4>Model Reasoning</h4>
                <p className="explanation-text">{results.explanation.model_reasoning}</p>
              </div>
              
              {results.explanation.explanation_text && (
                <div className="explanation-section">
                  <h4>Decision Process</h4>
                  <p className="explanation-text">{results.explanation.explanation_text}</p>
                </div>
              )}
              
              {Object.keys(results.explanation.feature_importance).length > 0 && (
                <div className="explanation-section">
                  <h4>Feature Importance</h4>
                  <div className="feature-importance">
                    {Object.entries(results.explanation.feature_importance).map(([feature, importance]) => (
                      <div key={feature} className="feature-item">
                        <span className="feature-name">{feature.replace(/_/g, ' ')}</span>
                        <div className="importance-bar">
                          <div 
                            className="importance-fill" 
                            style={{ width: `${importance * 100}%` }}
                          ></div>
                          <span className="importance-value">{(importance * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {Object.keys(results.explanation.confidence_breakdown).length > 0 && (
                <div className="explanation-section">
                  <h4>Confidence Breakdown</h4>
                  <div className="confidence-breakdown">
                    {Object.entries(results.explanation.confidence_breakdown).map(([component, confidence]) => (
                      <div key={component} className="confidence-item">
                        <span className="confidence-component">{component.replace(/_/g, ' ')}</span>
                        <span className="confidence-score">{(confidence * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <style>{`
        .intelligent-results {
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          overflow: hidden;
        }

        .results-header {
          background: #f8f9fa;
          padding: 20px;
          border-bottom: 1px solid #e0e0e0;
        }

        .model-info {
          display: flex;
          align-items: flex-start;
          gap: 15px;
          margin-bottom: 20px;
        }

        .model-icon {
          font-size: 2em;
        }

        .model-details h3 {
          margin: 0 0 5px 0;
          color: #333;
        }

        .model-description {
          color: #666;
          margin: 0 0 10px 0;
        }

        .processing-info {
          display: flex;
          gap: 20px;
          font-size: 0.9em;
          color: #777;
        }

        .confidence-scores {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .confidence-item {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .confidence-label {
          min-width: 120px;
          font-size: 0.9em;
          color: #555;
        }

        .confidence-bar {
          flex: 1;
          height: 20px;
          background: #e0e0e0;
          border-radius: 10px;
          position: relative;
          overflow: hidden;
        }

        .confidence-fill {
          height: 100%;
          background: linear-gradient(90deg, #4caf50, #8bc34a);
          transition: width 0.3s ease;
        }

        .confidence-value {
          position: absolute;
          right: 8px;
          top: 50%;
          transform: translateY(-50%);
          font-size: 0.8em;
          font-weight: bold;
          color: #333;
        }

        .results-tabs {
          display: flex;
          background: #f0f0f0;
          border-bottom: 1px solid #e0e0e0;
          overflow-x: auto;
        }

        .tab {
          background: none;
          border: none;
          padding: 12px 20px;
          cursor: pointer;
          white-space: nowrap;
          color: #666;
          font-weight: 500;
          transition: all 0.2s ease;
        }

        .tab:hover {
          background: #e8e8e8;
          color: #333;
        }

        .tab.active {
          background: white;
          color: #007bff;
          border-bottom: 2px solid #007bff;
        }

        .results-content {
          padding: 20px;
        }

        .overview-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
        }

        .overview-card {
          background: #f8f9fa;
          border-radius: 6px;
          padding: 15px;
        }

        .overview-card h4 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .summary-stats {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 15px;
        }

        .stat {
          text-align: center;
        }

        .stat-number {
          display: block;
          font-size: 2em;
          font-weight: bold;
          color: #007bff;
        }

        .stat-label {
          font-size: 0.9em;
          color: #666;
        }

        .entities-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
          gap: 15px;
        }

        .entity-card {
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          padding: 12px;
        }

        .entity-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 8px;
        }

        .entity-type {
          background: #e3f2fd;
          color: #1976d2;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 0.8em;
          font-weight: 500;
        }

        .entity-confidence {
          font-size: 0.9em;
          color: #666;
        }

        .entity-text {
          font-weight: 500;
          margin-bottom: 5px;
        }

        .entity-position {
          font-size: 0.8em;
          color: #999;
        }

        .treatments-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
          gap: 20px;
        }

        .treatment-card {
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          padding: 20px;
          background: #fafafa;
        }

        .treatment-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 15px;
          padding-bottom: 10px;
          border-bottom: 1px solid #e0e0e0;
        }

        .treatment-condition {
          margin: 0;
          color: #2c5530;
          font-size: 1.1em;
        }

        .treatment-confidence {
          background: #e8f5e8;
          color: #2c5530;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 0.8em;
          font-weight: 500;
        }

        .treatment-details {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .treatment-details > div {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .treatment-details strong {
          color: #333;
          font-size: 0.9em;
        }

        .treatment-details span {
          color: #555;
          padding-left: 10px;
        }

        .sanskrit-name {
          display: block;
          color: #7b1fa2;
          font-style: italic;
          font-size: 0.85em;
          padding-left: 10px;
          margin-top: 2px;
        }

        .evidence-badge {
          color: white;
          padding: 2px 8px;
          border-radius: 10px;
          font-size: 0.7em;
          font-weight: 500;
          text-transform: uppercase;
          display: inline-block;
          margin-left: 10px;
        }

        .knowledge-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 15px;
        }

        .knowledge-card {
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          padding: 15px;
          background: #fafafa;
        }

        .knowledge-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 10px;
        }

        .knowledge-header h5 {
          margin: 0;
          color: #333;
          font-size: 1em;
        }

        .knowledge-confidence {
          background: #e3f2fd;
          color: #1976d2;
          padding: 2px 6px;
          border-radius: 10px;
          font-size: 0.8em;
        }

        .knowledge-content {
          margin-bottom: 10px;
        }

        .knowledge-content p {
          margin: 5px 0;
          color: #555;
          line-height: 1.4;
        }

        .knowledge-source {
          font-size: 0.8em;
          color: #777;
          border-top: 1px solid #e0e0e0;
          padding-top: 8px;
        }

        .interactions-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 15px;
        }

        .interaction-card {
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          padding: 15px;
          background: #fff8e1;
        }

        .interaction-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .interaction-header h5 {
          margin: 0;
          color: #333;
        }

        .severity-badge {
          color: white;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 0.8em;
          font-weight: 500;
        }

        .interaction-details p {
          margin: 5px 0;
          color: #555;
        }

        .safety-content {
          max-width: 800px;
        }

        .safety-overview {
          display: flex;
          flex-direction: column;
          gap: 20px;
        }

        .risk-summary {
          display: flex;
          align-items: center;
          gap: 20px;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 6px;
        }

        .risk-level {
          color: white;
          padding: 8px 16px;
          border-radius: 20px;
          font-weight: bold;
          font-size: 0.9em;
        }

        .risk-score {
          font-size: 1.1em;
          font-weight: 500;
          color: #333;
        }

        .risk-factors {
          background: #f8f9fa;
          padding: 15px;
          border-radius: 6px;
        }

        .risk-factors h5 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .risk-factor {
          display: flex;
          justify-content: space-between;
          padding: 5px 0;
          border-bottom: 1px solid #e0e0e0;
        }

        .risk-factor:last-child {
          border-bottom: none;
        }

        .safety-recommendations {
          background: #e8f5e8;
          padding: 15px;
          border-radius: 6px;
        }

        .safety-recommendations h5 {
          margin: 0 0 10px 0;
          color: #2c5530;
        }

        .safety-recommendations ul {
          margin: 0;
          padding-left: 20px;
        }

        .safety-recommendations li {
          margin: 5px 0;
          color: #555;
        }

        .consultation-warning {
          background: #fff3cd;
          border: 1px solid #ffeaa7;
          color: #856404;
          padding: 10px;
          border-radius: 6px;
          font-weight: 500;
          text-align: center;
        }

        .routing-tab {
          padding: 20px;
        }

        .routing-info h3 {
          margin: 0 0 20px 0;
          color: #333;
        }

        .routing-details {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
          gap: 20px;
        }

        .routing-card {
          background: #f8f9fa;
          border-radius: 6px;
          padding: 15px;
          border-left: 4px solid #007bff;
        }

        .routing-card h4 {
          margin: 0 0 15px 0;
          color: #333;
        }

        .routing-card p {
          margin: 8px 0;
          color: #555;
        }

        .explanation-tab {
          padding: 20px;
        }

        .explanation-content h3 {
          margin: 0 0 20px 0;
          color: #333;
        }

        .explanation-section {
          margin-bottom: 25px;
          padding: 15px;
          background: #f8f9fa;
          border-radius: 6px;
        }

        .explanation-section h4 {
          margin: 0 0 15px 0;
          color: #333;
          font-size: 1.1em;
        }

        .explanation-text {
          line-height: 1.6;
          color: #555;
          margin: 0;
        }

        .feature-importance {
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .feature-item {
          display: flex;
          align-items: center;
          gap: 15px;
        }

        .feature-name {
          min-width: 120px;
          font-weight: 500;
          color: #333;
          text-transform: capitalize;
        }

        .importance-bar {
          flex: 1;
          height: 20px;
          background: #e0e0e0;
          border-radius: 10px;
          position: relative;
          overflow: hidden;
        }

        .importance-fill {
          height: 100%;
          background: linear-gradient(90deg, #4caf50, #8bc34a);
          transition: width 0.3s ease;
        }

        .importance-value {
          position: absolute;
          right: 8px;
          top: 50%;
          transform: translateY(-50%);
          font-size: 0.8em;
          font-weight: bold;
          color: #333;
        }

        .confidence-breakdown {
          display: flex;
          flex-direction: column;
          gap: 10px;
        }

        .confidence-item {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 8px 12px;
          background: white;
          border-radius: 4px;
          border-left: 3px solid #007bff;
        }

        .confidence-component {
          font-weight: 500;
          color: #333;
          text-transform: capitalize;
        }

        .confidence-score {
          font-weight: bold;
          color: #007bff;
        }
      `}</style>
    </div>
  )
}