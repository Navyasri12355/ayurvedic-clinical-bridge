/**
 * Explanation Viewer Component
 * 
 * Displays SHAP-based explanations for ML model predictions
 * with interactive visualizations and detailed insights.
 */

import React, { useState } from 'react'

interface WordExplanation {
  word: string
  importance: number
  contribution: 'positive' | 'negative' | 'neutral'
}

interface ExplanationData {
  available: boolean
  summary?: string
  explanation_summary?: string
  important_words?: WordExplanation[]
  word_importance?: WordExplanation[]  // Add support for backend field name
  confidence_assessment?: {
    confidence_available: boolean
    explanation_confidence: string
    confidence_summary: string
    metrics?: {
      significant_words_count: number
      total_words_analyzed: number
      prediction_confidence: number
    }
  }
}

interface Props {
  explanation: ExplanationData
  prediction?: {
    disease: string
    confidence: string
  }
}

export default function ExplanationViewer({ explanation, prediction }: Props) {
  const [showDetails, setShowDetails] = useState(false)
  const [activeTab, setActiveTab] = useState<'summary' | 'words' | 'confidence'>('summary')

  // Get word importance data from either field name
  const wordImportanceData = explanation.word_importance || explanation.important_words || []
  
  // Get summary from either field name
  const explanationSummary = explanation.explanation_summary || explanation.summary || ''

  if (!explanation?.available) {
    return (
      <div className="explanation-unavailable">
        <div className="unavailable-content">
          <span className="unavailable-icon">🔍</span>
          <h4>Explanation Not Available</h4>
          <p>Detailed explanation could not be generated for this prediction.</p>
        </div>
      </div>
    )
  }

  const getImportanceColor = (contribution: string, importance: number) => {
    const intensity = Math.min(Math.abs(importance) * 10, 1)
    
    switch (contribution) {
      case 'positive':
        return `rgba(46, 139, 87, ${0.1 + intensity * 0.2})` // Very light Sea Green background
      case 'negative':
        return `rgba(220, 20, 60, ${0.1 + intensity * 0.2})` // Very light Crimson background
      default:
        return `rgba(128, 128, 128, ${0.1 + intensity * 0.2})` // Very light Gray background
    }
  }

  const getBorderColor = (contribution: string, importance: number) => {
    const intensity = Math.min(Math.abs(importance) * 10, 1)
    
    switch (contribution) {
      case 'positive':
        return `rgba(46, 139, 87, ${0.6 + intensity * 0.4})` // Solid Sea Green border
      case 'negative':
        return `rgba(220, 20, 60, ${0.6 + intensity * 0.4})` // Solid Crimson border
      default:
        return `rgba(128, 128, 128, ${0.6 + intensity * 0.4})` // Solid Gray border
    }
  }

  const getImportanceIcon = (contribution: string) => {
    switch (contribution) {
      case 'positive':
        return '✅'
      case 'negative':
        return '❌'
      default:
        return '➖'
    }
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case 'high':
        return '#52c282ff'
      case 'medium':
        return '#eea449ff'
      case 'low':
        return '#d54461ff'
      default:
        return '#e7e2e2ff'
    }
  }

  return (
    <div className="explanation-viewer">
      <div className="explanation-header">
        <div className="header-content">
          <h3>🧠 AI Explanation</h3>
          <p>Understanding how the AI made this prediction</p>
        </div>
        <button 
          className="toggle-details"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {prediction && (
        <div className="prediction-context">
          <div className="prediction-info">
            <span className="prediction-label">Predicted:</span>
            <span className="prediction-value">{prediction.disease}</span>
            <span className="prediction-confidence">({prediction.confidence})</span>
          </div>
        </div>
      )}

      <div className="explanation-summary">
        <div className="summary-content">
          <span className="summary-icon">💡</span>
          <p>{explanationSummary}</p>
        </div>
      </div>

      {showDetails && (
        <div className="explanation-details">
          <div className="details-tabs">
            <button 
              className={`tab ${activeTab === 'summary' ? 'active' : ''}`}
              onClick={() => setActiveTab('summary')}
            >
              Summary
            </button>
            <button 
              className={`tab ${activeTab === 'words' ? 'active' : ''}`}
              onClick={() => setActiveTab('words')}
            >
              Word Analysis ({wordImportanceData?.length || 0})
            </button>
            {explanation.confidence_assessment?.confidence_available && (
              <button 
                className={`tab ${activeTab === 'confidence' ? 'active' : ''}`}
                onClick={() => setActiveTab('confidence')}
              >
                Confidence
              </button>
            )}
          </div>

          <div className="tab-content">
            {activeTab === 'summary' && (
              <div className="summary-tab">
                <div className="explanation-method">
                  <h4>📊 Analysis Method</h4>
                  <p>
                    This explanation uses <strong>SHAP (SHapley Additive exPlanations)</strong>, 
                    a state-of-the-art method for understanding AI model decisions. 
                    SHAP assigns each word an importance score that represents its contribution 
                    to the final prediction.
                  </p>
                </div>
                
                <div className="how-to-read">
                  <h4>📖 How to Read This</h4>
                  <ul>
                    <li><span style={{color: '#2E8B57'}}>✅ Green words</span> support the prediction</li>
                    <li><span style={{color: '#DC143C'}}>❌ Red words</span> work against the prediction</li>
                    <li><span style={{color: '#808080'}}>➖ Gray words</span> have neutral impact</li>
                    <li>Darker colors indicate stronger influence</li>
                  </ul>
                </div>
              </div>
            )}

            {activeTab === 'words' && (
              <div className="words-tab">
                <h4>🔤 Word Importance Analysis</h4>
                <div className="words-grid">
                  {wordImportanceData?.map((word, index) => (
                    <div 
                      key={index} 
                      className="word-item"
                      style={{
                        backgroundColor: getImportanceColor(word.contribution, word.importance),
                        border: `2px solid ${getBorderColor(word.contribution, word.importance)}`,
                        borderRadius: '8px'
                      }}
                    >
                      <div className="word-header">
                        <span className="word-icon">
                          {getImportanceIcon(word.contribution)}
                        </span>
                        <span className="word-text">"{word.word}"</span>
                      </div>
                      <div className="word-details">
                        <span className="importance-score">
                          Impact: {word.importance > 0 ? '+' : ''}{word.importance.toFixed(3)}
                        </span>
                        <span className="contribution-type">
                          {word.contribution === 'positive' ? 'Supports' : 
                           word.contribution === 'negative' ? 'Opposes' : 'Neutral'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                
                {(!wordImportanceData || wordImportanceData.length === 0) && (
                  <div className="no-words">
                    <p>No significant word importance data available.</p>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'confidence' && explanation.confidence_assessment?.confidence_available && (
              <div className="confidence-tab">
                <h4>🎯 Explanation Confidence</h4>
                
                <div className="confidence-overview">
                  <div className="confidence-badge">
                    <span 
                      className="confidence-level"
                      style={{
                        backgroundColor: getConfidenceColor(explanation.confidence_assessment.explanation_confidence),
                        color: 'white'
                      }}
                    >
                      {explanation.confidence_assessment.explanation_confidence.toUpperCase()} CONFIDENCE
                    </span>
                  </div>
                  <p>{explanation.confidence_assessment.confidence_summary}</p>
                </div>

                {explanation.confidence_assessment.metrics && (
                  <div className="confidence-metrics">
                    <h5>📈 Detailed Metrics</h5>
                    <div className="metrics-grid">
                      <div className="metric-item">
                        <span className="metric-label">Significant Words:</span>
                        <span className="metric-value">
                          {explanation.confidence_assessment.metrics.significant_words_count} / {explanation.confidence_assessment.metrics.total_words_analyzed}
                        </span>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">Prediction Confidence:</span>
                        <span className="metric-value">
                          {(explanation.confidence_assessment.metrics.prediction_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      <div className="explanation-disclaimer">
        <p>
          <strong>Note:</strong> This explanation shows how the AI model made its decision 
          based on the words in your input. It's designed to increase transparency and trust, 
          but should be considered alongside professional medical advice.
        </p>
      </div>

      <style>{`
        .explanation-viewer {
          background: #f8f9fa;
          border-radius: 8px;
          border: 1px solid #e0e0e0;
          margin-top: 20px;
          overflow: hidden;
        }

        .explanation-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .header-content h3 {
          margin: 0 0 5px 0;
          font-size: 1.3em;
        }

        .header-content p {
          margin: 0;
          opacity: 0.9;
          font-size: 0.9em;
        }

        .toggle-details {
          background: rgba(255, 255, 255, 0.2);
          border: 1px solid rgba(255, 255, 255, 0.3);
          color: white;
          padding: 8px 16px;
          border-radius: 20px;
          cursor: pointer;
          font-size: 0.9em;
          transition: all 0.2s ease;
        }

        .toggle-details:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .prediction-context {
          background: #e8f4fd;
          padding: 15px 20px;
          border-bottom: 1px solid #e0e0e0;
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

        .explanation-summary {
          padding: 20px;
          background: white;
        }

        .summary-content {
          display: flex;
          align-items: flex-start;
          gap: 15px;
        }

        .summary-icon {
          font-size: 1.5em;
          margin-top: 2px;
        }

        .summary-content p {
          margin: 0;
          line-height: 1.6;
          color: #333;
        }

        .explanation-details {
          border-top: 1px solid #e0e0e0;
        }

        .details-tabs {
          display: flex;
          background: #f0f0f0;
          border-bottom: 1px solid #e0e0e0;
        }

        .tab {
          background: none;
          border: none;
          padding: 12px 20px;
          cursor: pointer;
          color: #666;
          font-weight: 500;
          transition: all 0.2s ease;
          border-bottom: 2px solid transparent;
        }

        .tab:hover {
          background: #e8e8e8;
          color: #333;
        }

        .tab.active {
          background: white;
          color: #007bff;
          border-bottom-color: #007bff;
        }

        .tab-content {
          padding: 20px;
          background: white;
        }

        .explanation-method,
        .how-to-read {
          margin-bottom: 20px;
        }

        .explanation-method h4,
        .how-to-read h4 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .how-to-read ul {
          margin: 10px 0;
          padding-left: 20px;
        }

        .how-to-read li {
          margin-bottom: 5px;
        }

        .words-grid {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
          gap: 15px;
          margin-top: 15px;
        }

        .word-item {
          border-radius: 8px;
          padding: 15px;
          transition: transform 0.2s ease, box-shadow 0.2s ease;
          background: white;
        }

        .word-item:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .word-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
        }

        .word-icon {
          font-size: 1.1em;
        }

        .word-text {
          font-weight: 600;
          color: #333;
        }

        .word-details {
          display: flex;
          justify-content: space-between;
          font-size: 0.85em;
          color: #555;
        }

        .confidence-overview {
          margin-bottom: 20px;
        }

        .confidence-badge {
          margin-bottom: 10px;
        }

        .confidence-level {
          padding: 6px 12px;
          border-radius: 15px;
          font-size: 0.8em;
          font-weight: 600;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 15px;
          margin-top: 10px;
        }

        .metric-item {
          display: flex;
          justify-content: space-between;
          padding: 10px;
          background: #f8f9fa;
          border-radius: 4px;
        }

        .metric-label {
          font-weight: 500;
          color: #555;
        }

        .metric-value {
          font-weight: 600;
          color: #007bff;
        }

        .explanation-disclaimer {
          background: #fff3cd;
          border-top: 1px solid #ffeaa7;
          padding: 15px 20px;
        }

        .explanation-disclaimer p {
          margin: 0;
          font-size: 0.9em;
          color: #856404;
          line-height: 1.5;
        }

        .explanation-unavailable {
          background: #f8f9fa;
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          padding: 30px;
          text-align: center;
          margin-top: 20px;
        }

        .unavailable-content {
          color: #666;
        }

        .unavailable-icon {
          font-size: 2em;
          display: block;
          margin-bottom: 10px;
        }

        .unavailable-content h4 {
          margin: 0 0 10px 0;
          color: #333;
        }

        .unavailable-content p {
          margin: 0;
        }

        .no-words {
          text-align: center;
          padding: 20px;
          color: #666;
        }
      `}</style>
    </div>
  )
}