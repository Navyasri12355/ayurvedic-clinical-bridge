/**
 * Intelligent Query Form Component
 * 
 * This component provides a unified interface for querying the intelligent
 * routing system that automatically selects the best model based on user
 * type and query complexity.
 */

import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { 
  intelligentQueryService, 
  queryHelpers,
  IntelligentQueryResponse,
  ModelCapabilitiesResponse 
} from '../services/intelligentQueryService'

interface Props {
  onResults: (results: IntelligentQueryResponse) => void
  onError: (error: string) => void
  placeholder?: string
  queryType?: 'general' | 'medicine_mapping' | 'clinical' | 'safety' | 'treatment'
  showAdvancedOptions?: boolean
  onSubmit?: (query: string) => void
  loading?: boolean
}

export default function IntelligentQueryForm({ 
  onResults, 
  onError, 
  placeholder,
  queryType = 'general',
  showAdvancedOptions = false,
  onSubmit,
  loading: externalLoading = false
}: Props) {
  const { user, token } = useAuth()
  const [query, setQuery] = useState('')
  const [accuracy, setAccuracy] = useState<'balanced' | 'high'>('balanced')
  const [includeExplanation, setIncludeExplanation] = useState(true) // Default to true
  const [loading, setLoading] = useState(false)
  const [capabilities, setCapabilities] = useState<ModelCapabilitiesResponse | null>(null)

  const isLoading = loading || externalLoading

  // Set auth token when available
  useEffect(() => {
    if (token) {
      intelligentQueryService.setAuthToken(token)
    }
  }, [token])

  // Load capabilities on mount
  useEffect(() => {
    const loadCapabilities = async () => {
      try {
        const caps = await intelligentQueryService.getCapabilities()
        setCapabilities(caps)
      } catch (error) {
        console.warn('Failed to load capabilities:', error)
      }
    }
    loadCapabilities()
  }, [])

  // Auto-set accuracy based on user role and query type
  useEffect(() => {
    if (user?.role === 'qualified_practitioner') {
      if (queryType === 'clinical' || queryType === 'safety' || queryType === 'treatment') {
        setAccuracy('high')
      } else {
        setAccuracy('balanced')
      }
    } else {
      // General users always use balanced (BiLSTM-CRF)
      setAccuracy('balanced')
    }
  }, [user, queryType])

  const getDefaultPlaceholder = () => {
    switch (queryType) {
      case 'general':
        return 'Ask about Ayurvedic concepts, herbs, or general health information...'
      case 'medicine_mapping':
        return 'Find Ayurvedic alternatives for medications or treatments...'
      case 'clinical':
        return 'Clinical query about treatments, protocols, or analysis...'
      case 'safety':
        return 'Analyze herb-drug interactions or safety concerns...'
      case 'treatment':
        return 'Request comprehensive treatment protocols for conditions...'
      default:
        return 'Enter your query...'
    }
  }

  const createQueryRequest = () => {
    switch (queryType) {
      case 'general':
        return queryHelpers.createGeneralQuery(query)
      case 'medicine_mapping':
        return queryHelpers.createMedicineMappingQuery(query)
      case 'clinical':
        return queryHelpers.createClinicalQuery(query)
      case 'treatment':
        // Extract condition from query or use the whole query
        const condition = query.toLowerCase().includes('treatment for') 
          ? query.split('treatment for')[1]?.trim() 
          : query
        return queryHelpers.createTreatmentQuery(condition)
      default:
        return {
          query,
          preferred_accuracy: accuracy,
          include_routing_info: true // Always include routing info
        }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    // If external onSubmit is provided, use it instead
    if (onSubmit) {
      onSubmit(query)
      return
    }

    setLoading(true)

    try {
      const request = createQueryRequest()
      request.preferred_accuracy = accuracy
      request.include_routing_info = true // Always include routing info
      request.include_explanation = includeExplanation

      const results = await intelligentQueryService.processQuery(request)
      onResults(results)
    } catch (error) {
      onError(error instanceof Error ? error.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const getAccuracyDescription = (acc: string) => {
    switch (acc) {
      case 'balanced':
        return 'Balanced (~0.4s) - BiLSTM-CRF for entity recognition'
      case 'high':
        return 'High accuracy (~1.8s) - BioBERT for clinical analysis'
      default:
        return ''
    }
  }

  const getExpectedModel = () => {
    if (!user) return 'BiLSTM-CRF (anonymous)'
    
    const isPractitioner = user.role === 'qualified_practitioner'
    
    if (accuracy === 'high') return 'BioBERT'
    if (queryType === 'clinical' || queryType === 'safety' || queryType === 'treatment') {
      return isPractitioner ? 'BioBERT (Clinical)' : 'BiLSTM-CRF → BioBERT'
    }
    return 'BiLSTM-CRF'
  }

  return (
    <div className="intelligent-query-form">
      <form onSubmit={handleSubmit} className="query-form">
        <div className="form-group">
          <label htmlFor="intelligent-query">
            Query ({queryType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}):
          </label>
          <textarea
            id="intelligent-query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={placeholder || getDefaultPlaceholder()}
            rows={3}
            disabled={isLoading}
            className="query-textarea"
          />
        </div>

        {showAdvancedOptions && (
          <div className="advanced-options">
            <div className="form-group">
              <label htmlFor="accuracy-select">Accuracy Preference:</label>
              <select
                id="accuracy-select"
                value={accuracy}
                onChange={(e) => setAccuracy(e.target.value as 'balanced' | 'high')}
                disabled={isLoading}
              >
                <option value="balanced">Balanced</option>
                <option value="high">High Accuracy</option>
              </select>
              <small className="accuracy-description">
                {getAccuracyDescription(accuracy)}
              </small>
            </div>

            <div className="checkbox-inline">
              <div className="checkbox-line">
                <input
                  type="checkbox"
                  id="include-explanation"
                  checked={includeExplanation}
                  onChange={(e) => setIncludeExplanation(e.target.checked)}
                  disabled={isLoading}
                />
                <label htmlFor="include-explanation">Include AI explanation (SHAP analysis)</label>
              </div>
              <small className="explanation-note">
                Shows how the AI model made its decision (may increase processing time)
              </small>
            </div>

            <div className="routing-preview">
              <small>
                <strong>Expected Model:</strong> {getExpectedModel()}
              </small>
            </div>
          </div>
        )}

        <div className="form-actions">
          <button 
            type="submit" 
            disabled={isLoading || !query.trim()}
            className="submit-button"
          >
            {isLoading ? (
              <>
                <span className="loading-spinner"></span>
                Processing...
              </>
            ) : (
              'Submit Query'
            )}
          </button>

          {capabilities && (
            <div className="capabilities-info">
              <small>
                Available Models: {capabilities.available_models.join(', ')}
              </small>
            </div>
          )}
        </div>
      </form>

      {user && (
        <div className="user-context">
          <div className="user-info">
            <span className="user-role">
              Role: {user.role === 'qualified_practitioner' ? 'Practitioner' : 'General User'}
            </span>
            {user.role === 'qualified_practitioner' && user.credentials?.verification_status && (
              <span className="verification-status verified">
                ✓ Verified
              </span>
            )}
          </div>
        </div>
      )}

      <style>{`
        .intelligent-query-form {
          background: #f8f9fa;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
        }

        .query-form {
          display: flex;
          flex-direction: column;
          gap: 15px;
        }

        .form-group {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .query-textarea {
          min-height: 80px;
          padding: 10px;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-family: inherit;
          resize: vertical;
        }

        .advanced-options {
          background: #fff;
          border: 1px solid #e0e0e0;
          border-radius: 6px;
          padding: 15px;
          margin-top: 10px;
        }

        .accuracy-description {
          color: #666;
          font-style: italic;
        }

        .explanation-note {
          color: #666;
          font-size: 0.85em;
          margin-top: 4px;
          font-style: italic;
        }

        .checkbox-inline {
          margin-top: 10px;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .checkbox-line {
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .checkbox-line input[type="checkbox"] {
          margin: 0;
          flex-shrink: 0;
        }

        .checkbox-line label {
          margin: 0;
          font-weight: normal;
          cursor: pointer;
        }

        .routing-preview {
          background: #e8f4f8;
          padding: 8px;
          border-radius: 4px;
          margin-top: 10px;
        }

        .form-actions {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 15px;
        }

        .submit-button {
          background: #007bff;
          color: white;
          border: none;
          padding: 10px 20px;
          border-radius: 4px;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 8px;
          font-weight: 500;
        }

        .submit-button:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        .loading-spinner {
          width: 16px;
          height: 16px;
          border: 2px solid #fff;
          border-top: 2px solid transparent;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .capabilities-info {
          color: #666;
        }

        .user-context {
          margin-top: 15px;
          padding-top: 15px;
          border-top: 1px solid #e0e0e0;
        }

        .user-info {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .user-role {
          background: #e3f2fd;
          color: #1976d2;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
        }

        .verification-status.verified {
          background: #e8f5e8;
          color: #2e7d32;
          padding: 4px 8px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
        }
      `}</style>
    </div>
  )
}