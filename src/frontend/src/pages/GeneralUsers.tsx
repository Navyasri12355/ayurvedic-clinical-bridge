import React, { useState } from 'react'
import IntelligentQueryForm from '../components/IntelligentQueryForm'
import IntelligentResults from '../components/IntelligentResults'
import { IntelligentQueryResponse } from '../services/intelligentQueryService'

export default function GeneralUsers() {
  const [results, setResults] = useState<IntelligentQueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleResults = (queryResults: IntelligentQueryResponse) => {
    setResults(queryResults)
    setError(null)
  }

  const handleError = (errorMessage: string) => {
    setError(errorMessage)
    setResults(null)
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
        
        <div className="system-info">
          <h3>🧠 Intelligent System Features</h3>
          <div className="features-grid">
            <div className="feature-card">
              <h4>⚡ Fast Responses</h4>
              <p>Knowledge base semantic search provides instant answers to general questions about Ayurvedic concepts and herbs.</p>
            </div>
            <div className="feature-card">
              <h4>🎯 Smart Routing</h4>
              <p>Queries are automatically routed to the most appropriate model based on complexity and content type.</p>
            </div>
            <div className="feature-card">
              <h4>📚 Comprehensive Knowledge</h4>
              <p>Access to structured knowledge base with evidence levels, dosage guidelines, and preparation methods.</p>
            </div>
          </div>
        </div>
      </div>

      <div className="query-section">
        <IntelligentQueryForm
          onResults={handleResults}
          onError={handleError}
          queryType="general"
          placeholder="Ask about Ayurvedic concepts, herbs, benefits, or general health information..."
          showAdvancedOptions={true}
        />
      </div>

      {error && (
        <div className="error-message">
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      )}

      {results && (
        <div className="results-section">
          <IntelligentResults 
            results={results} 
            showRoutingInfo={true}
          />
          
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

      <style jsx="true">{`
        .general-users-page {
          max-width: 1200px;
          margin: 0 auto;
          padding: 20px;
        }

        .header {
          margin-bottom: 30px;
        }

        .header h1 {
          color: #2c3e50;
          margin-bottom: 20px;
        }

        .disclaimer {
          background: #fff3cd;
          border: 1px solid #ffeaa7;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 20px;
        }

        .disclaimer h3 {
          color: #856404;
          margin: 0 0 10px 0;
        }

        .disclaimer p {
          color: #856404;
          margin: 0;
        }

        .system-info {
          background: #e8f4f8;
          border-radius: 8px;
          padding: 20px;
          margin-bottom: 20px;
        }

        .system-info h3 {
          color: #1976d2;
          margin: 0 0 15px 0;
        }

        .features-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 15px;
        }

        .feature-card {
          background: white;
          border-radius: 6px;
          padding: 15px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .feature-card h4 {
          color: #1976d2;
          margin: 0 0 8px 0;
          font-size: 1em;
        }

        .feature-card p {
          color: #666;
          margin: 0;
          font-size: 0.9em;
          line-height: 1.4;
        }

        .query-section {
          margin-bottom: 30px;
        }

        .error-message {
          background: #f8d7da;
          border: 1px solid #f5c6cb;
          border-radius: 8px;
          padding: 15px;
          margin-bottom: 20px;
        }

        .error-message h3 {
          color: #721c24;
          margin: 0 0 10px 0;
        }

        .error-message p {
          color: #721c24;
          margin: 0;
        }

        .results-section {
          margin-bottom: 30px;
        }

        .safety-reminder {
          background: #d1ecf1;
          border: 1px solid #bee5eb;
          border-radius: 8px;
          padding: 20px;
          margin-top: 30px;
        }

        .safety-reminder h3 {
          color: #0c5460;
          margin: 0 0 15px 0;
        }

        .safety-reminder p {
          color: #0c5460;
          margin: 0 0 10px 0;
        }

        .safety-reminder ul {
          color: #0c5460;
          margin: 0;
          padding-left: 20px;
        }

        .safety-reminder li {
          margin-bottom: 5px;
        }
      `}</style>
    </div>
  )
}