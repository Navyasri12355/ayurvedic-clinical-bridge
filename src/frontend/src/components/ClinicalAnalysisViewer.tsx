import React, { useState } from 'react';

/**
 * Clinical Analysis Results Component
 * Displays LLM-enhanced clinical analysis with reasoning chain
 */

interface ClinicalAnalysis {
  clinical_summary: string;
  key_findings: string[];
  recommendations: string[];
  safety_concerns: string[];
  follow_up_questions: string[];
  confidence_level: 'high' | 'moderate' | 'low';
  reasoning_chain: string;
}

interface ClinicalAnalysisViewerProps {
  analysis: ClinicalAnalysis;
  query: string;
  loading?: boolean;
}

const ClinicalAnalysisViewer: React.FC<ClinicalAnalysisViewerProps> = ({
  analysis,
  query,
  loading = false,
}) => {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    summary: true,
    findings: true,
    recommendations: true,
    safety: true,
    reasoning: false,
  });

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'high':
        return '#10b981';
      case 'moderate':
        return '#f59e0b';
      case 'low':
        return '#ef4444';
      default:
        return '#6b7280';
    }
  };

  const getConfidenceIcon = (level: string) => {
    switch (level) {
      case 'high':
        return '✅';
      case 'moderate':
        return '⚠️';
      case 'low':
        return '❓';
      default:
        return 'ℹ️';
    }
  };

  if (loading) {
    return (
      <div className="clinical-analysis-viewer loading">
        <div className="loader">
          <div className="spinner"></div>
          <p>Performing clinical analysis...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="clinical-analysis-viewer">
      <div className="analysis-header">
        <h2>Clinical Analysis Report</h2>
        <div className="confidence-badge" style={{ borderLeft: `4px solid ${getConfidenceColor(analysis.confidence_level)}` }}>
          <span className="icon">{getConfidenceIcon(analysis.confidence_level)}</span>
          <span className="text">
            Confidence: <strong>{analysis.confidence_level.toUpperCase()}</strong>
          </span>
        </div>
      </div>

      {/* Query Context */}
      <div className="query-context">
        <strong>Query:</strong>
        <p>{query}</p>
      </div>

      {/* Clinical Summary */}
      <div className="analysis-section">
        <div
          className="section-header"
          onClick={() => toggleSection('summary')}
        >
          <span className="toggle-icon">
            {expandedSections.summary ? '▼' : '▶'}
          </span>
          <h3>📋 Clinical Summary</h3>
        </div>
        {expandedSections.summary && (
          <div className="section-content">
            <p className="summary-text">{analysis.clinical_summary}</p>
          </div>
        )}
      </div>

      {/* Key Findings */}
      <div className="analysis-section">
        <div
          className="section-header"
          onClick={() => toggleSection('findings')}
        >
          <span className="toggle-icon">
            {expandedSections.findings ? '▼' : '▶'}
          </span>
          <h3>🔍 Key Findings</h3>
        </div>
        {expandedSections.findings && (
          <div className="section-content">
            <ul className="findings-list">
              {analysis.key_findings.map((finding, idx) => (
                <li key={idx}>
                  <span className="bullet">•</span>
                  {finding}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Recommendations */}
      <div className="analysis-section">
        <div
          className="section-header"
          onClick={() => toggleSection('recommendations')}
        >
          <span className="toggle-icon">
            {expandedSections.recommendations ? '▼' : '▶'}
          </span>
          <h3>💊 Recommendations</h3>
        </div>
        {expandedSections.recommendations && (
          <div className="section-content">
            <ul className="recommendations-list">
              {analysis.recommendations.map((rec, idx) => (
                <li key={idx} className="recommendation-item">
                  <span className="number">{idx + 1}</span>
                  <span className="text">{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Safety Concerns */}
      {analysis.safety_concerns.length > 0 && (
        <div className="analysis-section safety-section">
          <div
            className="section-header"
            onClick={() => toggleSection('safety')}
          >
            <span className="toggle-icon">
              {expandedSections.safety ? '▼' : '▶'}
            </span>
            <h3>⚠️ Safety Concerns</h3>
          </div>
          {expandedSections.safety && (
            <div className="section-content">
              <ul className="safety-list">
                {analysis.safety_concerns.map((concern, idx) => (
                  <li key={idx} className="safety-item">
                    {concern}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Follow-up Questions */}
      {analysis.follow_up_questions.length > 0 && (
        <div className="analysis-section">
          <div
            className="section-header"
            onClick={() => toggleSection('followup')}
          >
            <span className="toggle-icon">
              {expandedSections.followup ? '▼' : '▶'}
            </span>
            <h3>❓ Follow-up Questions</h3>
          </div>
          {expandedSections.followup && (
            <div className="section-content">
              <ul className="followup-list">
                {analysis.follow_up_questions.map((q, idx) => (
                  <li key={idx}>{q}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Reasoning Chain */}
      <div className="analysis-section">
        <div
          className="section-header"
          onClick={() => toggleSection('reasoning')}
        >
          <span className="toggle-icon">
            {expandedSections.reasoning ? '▼' : '▶'}
          </span>
          <h3>🧠 Clinical Reasoning Chain</h3>
        </div>
        {expandedSections.reasoning && (
          <div className="section-content">
            <div className="reasoning-box">
              <p>{analysis.reasoning_chain}</p>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .clinical-analysis-viewer {
          background: white;
          border-radius: 12px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          overflow: hidden;
          max-width: 900px;
          margin: 20px auto;
        }

        .clinical-analysis-viewer.loading {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 300px;
        }

        .loader {
          text-align: center;
        }

        .spinner {
          width: 50px;
          height: 50px;
          border: 4px solid #e5e7eb;
          border-top: 4px solid #667eea;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
          margin: 0 auto 20px;
        }

        @keyframes spin {
          to {
            transform: rotate(360deg);
          }
        }

        .analysis-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 24px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .analysis-header h2 {
          margin: 0;
          font-size: 24px;
        }

        .confidence-badge {
          display: flex;
          align-items: center;
          gap: 8px;
          background: rgba(255, 255, 255, 0.2);
          padding: 8px 16px;
          border-radius: 8px;
        }

        .confidence-badge .icon {
          font-size: 18px;
        }

        .query-context {
          background: #f3f4f6;
          padding: 16px 24px;
          border-bottom: 1px solid #e5e7eb;
        }

        .query-context strong {
          color: #374151;
        }

        .query-context p {
          margin: 8px 0 0 0;
          color: #555;
          font-style: italic;
        }

        .analysis-section {
          border-bottom: 1px solid #e5e7eb;
        }

        .analysis-section:last-child {
          border-bottom: none;
        }

        .analysis-section.safety-section .section-header {
          background: #fef2f2;
          border-left: 4px solid #ef4444;
        }

        .section-header {
          padding: 16px 24px;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 12px;
          transition: background 0.2s;
        }

        .section-header:hover {
          background: #f9fafb;
        }

        .toggle-icon {
          font-size: 12px;
          color: #667eea;
          user-select: none;
        }

        .section-header h3 {
          margin: 0;
          font-size: 16px;
          color: #1f2937;
        }

        .section-content {
          padding: 16px 24px;
          background: #fafbfc;
          animation: slideDown 0.2s ease-out;
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .summary-text {
          line-height: 1.6;
          color: #374151;
          margin: 0;
        }

        .findings-list,
        .recommendations-list,
        .safety-list,
        .followup-list {
          list-style: none;
          margin: 0;
          padding: 0;
        }

        .findings-list li,
        .followup-list li {
          padding: 8px 0;
          color: #374151;
          line-height: 1.5;
        }

        .findings-list .bullet {
          color: #667eea;
          font-weight: bold;
          margin-right: 8px;
        }

        .recommendation-item {
          display: flex;
          gap: 12px;
          padding: 12px;
          background: white;
          border-radius: 6px;
          margin-bottom: 8px;
          border-left: 3px solid #10b981;
        }

        .recommendation-item .number {
          background: #10b981;
          color: white;
          width: 24px;
          height: 24px;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: bold;
          flex-shrink: 0;
        }

        .recommendation-item .text {
          color: #374151;
          line-height: 1.5;
        }

        .safety-item {
          padding: 12px;
          background: white;
          border-radius: 6px;
          margin-bottom: 8px;
          border-left: 3px solid #ef4444;
          color: #374151;
          line-height: 1.5;
        }

        .reasoning-box {
          background: white;
          padding: 16px;
          border-radius: 6px;
          border-left: 3px solid #667eea;
          font-family: 'Monaco', 'Courier New', monospace;
          font-size: 13px;
          color: #374151;
          max-height: 400px;
          overflow-y: auto;
          line-height: 1.6;
        }

        .reasoning-box p {
          margin: 0;
          white-space: pre-wrap;
          word-wrap: break-word;
        }

        @media (max-width: 768px) {
          .analysis-header {
            flex-direction: column;
            gap: 12px;
            align-items: flex-start;
          }

          .analysis-header h2 {
            font-size: 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default ClinicalAnalysisViewer;
