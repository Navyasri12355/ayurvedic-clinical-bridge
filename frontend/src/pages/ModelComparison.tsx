import { useState, useEffect } from 'react'

interface ModelMetrics {
  model_name: string
  model_type: 'rnn' | 'lstm' | 'gru' | 'bilstm' | 'transformer'
  latency_ms: number
  memory_usage_mb: number
  training_loss: number[]
  validation_loss: number[]
  epochs: number[]
  throughput_samples_per_sec: number
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  parameters_count: number
  training_time_hours: number
  training_time_seconds?: number
  inference_time_ms: number
  gpu_memory_mb?: number
  cpu_usage_percent?: number
}

interface ComparisonData {
  models: ModelMetrics[]
  timestamp: string
  evaluation_dataset: string
}

interface TrainingStatus {
  training_in_progress: boolean
  active_training_threads: number
  last_update: string
}

export default function ModelComparison() {
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null)
  const [loading, setLoading] = useState(false)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedMetric, setSelectedMetric] = useState<string>('latency_ms')

  const backendUrl = import.meta.env?.VITE_BACKEND_URL || 'http://localhost:8000'

  useEffect(() => {
    fetchComparisonData()
    checkTrainingStatus()
    
    // Poll training status every 10 seconds
    const statusInterval = setInterval(checkTrainingStatus, 10000)
    return () => clearInterval(statusInterval)
  }, [])

  const fetchComparisonData = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${backendUrl}/api/models/comparison`)
      
      if (response.status === 404) {
        setError('No training data available. Please click "Trigger Training" to train models and generate metrics.')
        setLoading(false)
        return
      }
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setComparisonData(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model comparison data')
      console.error('Failed to fetch comparison data:', err)
    } finally {
      setLoading(false)
    }
  }

  const checkTrainingStatus = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/models/training-status`)
      
      if (response.ok) {
        const status = await response.json()
        setTrainingStatus(status)
      }
    } catch (err) {
      console.error('Failed to check training status:', err)
    }
  }

  const triggerTraining = async () => {
    try {
      setLoading(true)
      const response = await fetch(`${backendUrl}/api/models/trigger-training`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const result = await response.json()
        alert(`Training started! ${result.message}`)
        
        // Start polling for updates
        const pollInterval = setInterval(async () => {
          await checkTrainingStatus()
          await fetchComparisonData()
          
          if (!trainingStatus?.training_in_progress) {
            clearInterval(pollInterval)
          }
        }, 5000)
        
      } else {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
        alert(`Failed to start training: ${error.detail || 'Unknown error'}`)
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred'
      alert(`Error starting training: ${errorMessage}`)
    } finally {
      setLoading(false)
    }
  }

  const getMetricValue = (model: ModelMetrics, metric: string): number => {
    return (model as any)[metric] || 0
  }

  const formatMetricValue = (value: number, metric: string): string => {
    if (metric.includes('_ms')) return `${value.toFixed(1)} ms`
    if (metric.includes('_mb')) return `${value.toFixed(0)} MB`
    if (metric.includes('_sec')) return `${value.toFixed(0)}/sec`
    if (metric.includes('_hours')) {
      // Show in seconds if less than 1 minute, minutes if less than 1 hour, otherwise hours
      const totalSeconds = value * 3600
      if (totalSeconds < 60) {
        return `${totalSeconds.toFixed(1)} sec`
      } else if (totalSeconds < 3600) {
        return `${(totalSeconds / 60).toFixed(1)} min`
      } else {
        return `${value.toFixed(1)} hrs`
      }
    }
    if (metric.includes('_percent')) return `${value.toFixed(1)}%`
    if (['accuracy', 'precision', 'recall', 'f1_score'].includes(metric)) {
      return `${(value * 100).toFixed(1)}%`
    }
    if (metric === 'parameters_count') return `${(value / 1000000).toFixed(1)}M`
    return value.toFixed(3)
  }

  const getBestModel = (metric: string): string => {
    if (!comparisonData) return ''
    
    const isLowerBetter = ['latency_ms', 'memory_usage_mb', 'training_time_hours', 'inference_time_ms'].includes(metric)
    
    let bestModel = comparisonData.models[0]
    let bestValue = getMetricValue(bestModel, metric)
    
    comparisonData.models.forEach(model => {
      const value = getMetricValue(model, metric)
      if (isLowerBetter ? value < bestValue : value > bestValue) {
        bestValue = value
        bestModel = model
      }
    })
    
    return bestModel.model_name
  }

  const renderLatencyVsPerformanceChart = () => {
    if (!comparisonData) return null

    const models = comparisonData.models
    const colors = {
      'rnn': '#9333ea',
      'lstm': '#f59e0b',
      'gru': '#06b6d4',
      'bilstm': '#3b82f6',
      'transformer': '#ef4444'
    }

    // Calculate chart dimensions
    const maxLatency = Math.max(...models.map(m => m.latency_ms))
    const minLatency = Math.min(...models.map(m => m.latency_ms))
    const maxAccuracy = Math.max(...models.map(m => m.accuracy))
    const minAccuracy = Math.min(...models.map(m => m.accuracy))
    
    const latencyRange = maxLatency - minLatency
    const accuracyRange = maxAccuracy - minAccuracy

    return (
      <div className="latency-performance-chart">
        <h3>Performance vs Latency Trade-off</h3>
        <p className="chart-description">
          This chart shows the relationship between model accuracy and inference latency. 
          Models in the top-left corner offer the best balance of high accuracy and low latency.
        </p>
        <div className="chart-container">
          <svg width="800" height="400" style={{ background: '#f9fafb', borderRadius: '8px' }}>
            {/* Grid lines */}
            {[0, 1, 2, 3, 4].map(i => (
              <g key={`grid-${i}`}>
                <line
                  x1="80"
                  y1={80 + i * 60}
                  x2="750"
                  y2={80 + i * 60}
                  stroke="#e5e7eb"
                  strokeWidth="1"
                />
                <line
                  x1={80 + i * 167.5}
                  y1="80"
                  x2={80 + i * 167.5}
                  y2="320"
                  stroke="#e5e7eb"
                  strokeWidth="1"
                />
              </g>
            ))}

            {/* Axes */}
            <line x1="80" y1="320" x2="750" y2="320" stroke="#374151" strokeWidth="2" />
            <line x1="80" y1="80" x2="80" y2="320" stroke="#374151" strokeWidth="2" />

            {/* Axis labels */}
            <text x="400" y="360" textAnchor="middle" fontSize="14" fill="#374151" fontWeight="bold">
              Latency (ms) - Lower is Better
            </text>
            <text x="30" y="200" textAnchor="middle" fontSize="14" fill="#374151" fontWeight="bold" transform="rotate(-90 30 200)">
              Accuracy (%) - Higher is Better
            </text>

            {/* Y-axis scale */}
            {[0, 1, 2, 3, 4].map(i => {
              const value = minAccuracy + (accuracyRange * (4 - i) / 4)
              return (
                <text key={`y-${i}`} x="70" y={85 + i * 60} textAnchor="end" fontSize="12" fill="#6b7280">
                  {(value * 100).toFixed(0)}%
                </text>
              )
            })}

            {/* X-axis scale */}
            {[0, 1, 2, 3, 4].map(i => {
              const value = minLatency + (latencyRange * i / 4)
              return (
                <text key={`x-${i}`} x={80 + i * 167.5} y="340" textAnchor="middle" fontSize="12" fill="#6b7280">
                  {value.toFixed(0)}
                </text>
              )
            })}

            {/* Plot points */}
            {models.map((model, index) => {
              const x = 80 + ((model.latency_ms - minLatency) / latencyRange) * 670
              const y = 320 - ((model.accuracy - minAccuracy) / accuracyRange) * 240
              const color = colors[model.model_type]

              return (
                <g key={index}>
                  {/* Point */}
                  <circle
                    cx={x}
                    cy={y}
                    r="8"
                    fill={color}
                    stroke="#fff"
                    strokeWidth="2"
                    style={{ cursor: 'pointer' }}
                  >
                    <title>
                      {model.model_name}
                      {'\n'}Accuracy: {(model.accuracy * 100).toFixed(1)}%
                      {'\n'}Latency: {model.latency_ms.toFixed(1)}ms
                      {'\n'}F1 Score: {(model.f1_score * 100).toFixed(1)}%
                    </title>
                  </circle>
                  
                  {/* Label */}
                  <text
                    x={x}
                    y={y - 15}
                    textAnchor="middle"
                    fontSize="11"
                    fill={color}
                    fontWeight="600"
                  >
                    {model.model_name.split(' ')[0]}
                  </text>
                </g>
              )
            })}

            {/* Legend */}
            <g transform="translate(600, 30)">
              <text x="0" y="0" fontSize="12" fontWeight="bold" fill="#374151">Model Types</text>
              {Object.entries(colors).map(([type, color], i) => (
                <g key={type} transform={`translate(0, ${20 + i * 20})`}>
                  <circle cx="5" cy="0" r="5" fill={color} />
                  <text x="15" y="4" fontSize="11" fill="#6b7280">
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </text>
                </g>
              ))}
            </g>

            {/* Optimal zone indicator */}
            <rect
              x="80"
              y="80"
              width="200"
              height="100"
              fill="#10b981"
              opacity="0.1"
              stroke="#10b981"
              strokeWidth="1"
              strokeDasharray="5,5"
            />
            <text x="180" y="100" textAnchor="middle" fontSize="10" fill="#059669" fontWeight="600">
              Optimal Zone
            </text>
            <text x="180" y="115" textAnchor="middle" fontSize="9" fill="#059669">
              (High Accuracy, Low Latency)
            </text>
          </svg>
        </div>

        <div className="performance-insights">
          <div className="insight-item">
            <strong>🎯 Best Balance:</strong> 
            <span>
              {models.reduce((best, model) => {
                const score = model.accuracy / (model.latency_ms / 100)
                const bestScore = best.accuracy / (best.latency_ms / 100)
                return score > bestScore ? model : best
              }).model_name}
            </span>
          </div>
          <div className="insight-item">
            <strong>⚡ Fastest:</strong> 
            <span>{models.reduce((fastest, model) => model.latency_ms < fastest.latency_ms ? model : fastest).model_name}</span>
          </div>
          <div className="insight-item">
            <strong>🎓 Most Accurate:</strong> 
            <span>{models.reduce((best, model) => model.accuracy > best.accuracy ? model : best).model_name}</span>
          </div>
        </div>
      </div>
    )
  }

  if (loading) {
    return <div className="loading">Loading model comparison data...</div>
  }

  return (
    <div className="model-comparison-page">
      <div className="header">
        <h1>Model Performance Comparison</h1>
        <p>Comparative analysis of different architectures for medical NER</p>
        {error && <div className="info-message">{error}</div>}
        
        <div className="header-actions">
          <div className="training-controls">
            <button 
              className={`training-button ${trainingStatus?.training_in_progress ? 'training-active' : ''}`}
              onClick={triggerTraining}
              disabled={loading || trainingStatus?.training_in_progress}
            >
              {trainingStatus?.training_in_progress ? (
                <>
                  <span className="spinner">⟳</span>
                  Training in Progress...
                </>
              ) : (
                <>
                  🚀 Start New Training
                </>
              )}
            </button>
            
            {trainingStatus?.training_in_progress && (
              <div className="training-status">
                <p>Training {trainingStatus.active_training_threads} model(s)...</p>
                <p>Estimated time: ~5 minutes</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {comparisonData && (
        <>
          <div className="comparison-controls">
            <div className="metric-selector">
              <label htmlFor="metric-select">Compare by:</label>
              <select 
                id="metric-select"
                value={selectedMetric} 
                onChange={(e) => setSelectedMetric(e.target.value)}
              >
                <option value="latency_ms">Latency (ms)</option>
                <option value="memory_usage_mb">Memory Usage (MB)</option>
                <option value="throughput_samples_per_sec">Throughput (samples/sec)</option>
                <option value="accuracy">Accuracy</option>
                <option value="precision">Precision</option>
                <option value="recall">Recall</option>
                <option value="f1_score">F1 Score</option>
                <option value="parameters_count">Parameters Count</option>
                <option value="training_time_hours">Training Time (hours)</option>
                <option value="inference_time_ms">Inference Time (ms)</option>
              </select>
            </div>
            <div className="dataset-info">
              <p><strong>Evaluation Dataset:</strong> {comparisonData.evaluation_dataset}</p>
              <p><strong>Last Updated:</strong> {new Date(comparisonData.timestamp).toLocaleString()}</p>
            </div>
          </div>

          <div className="metrics-comparison">
            <h2>Performance Metrics Comparison</h2>
            <div className="metrics-table">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Type</th>
                    <th>Latency (ms)</th>
                    <th>Memory (MB)</th>
                    <th>Throughput</th>
                    <th>Accuracy</th>
                    <th>F1 Score</th>
                    <th>Parameters</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonData.models.map((model) => (
                    <tr key={model.model_name} className={getBestModel(selectedMetric) === model.model_name ? 'best-model' : ''}>
                      <td><strong>{model.model_name}</strong></td>
                      <td>
                        <span className={`model-type ${model.model_type}`}>
                          {model.model_type.toUpperCase()}
                        </span>
                      </td>
                      <td>{formatMetricValue(model.latency_ms, 'latency_ms')}</td>
                      <td>{formatMetricValue(model.memory_usage_mb, 'memory_usage_mb')}</td>
                      <td>{formatMetricValue(model.throughput_samples_per_sec, 'throughput_samples_per_sec')}</td>
                      <td>{formatMetricValue(model.accuracy, 'accuracy')}</td>
                      <td>{formatMetricValue(model.f1_score, 'f1_score')}</td>
                      <td>{formatMetricValue(model.parameters_count, 'parameters_count')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Latency vs Performance Trade-off Chart */}
          {renderLatencyVsPerformanceChart()}

          <div className="detailed-metrics">
            <h2>Detailed Performance Analysis</h2>
            <div className="metrics-grid">
              {comparisonData.models.map((model) => (
                <div key={model.model_name} className="model-card">
                  <h3>{model.model_name}</h3>
                  <div className="model-metrics">
                    <div className="metric-group">
                      <h4>Performance</h4>
                      <p>Accuracy: {formatMetricValue(model.accuracy, 'accuracy')}</p>
                      <p>Precision: {formatMetricValue(model.precision, 'precision')}</p>
                      <p>Recall: {formatMetricValue(model.recall, 'recall')}</p>
                      <p>F1 Score: {formatMetricValue(model.f1_score, 'f1_score')}</p>
                    </div>
                    <div className="metric-group">
                      <h4>Efficiency</h4>
                      <p>Latency: {formatMetricValue(model.latency_ms, 'latency_ms')}</p>
                      <p>Throughput: {formatMetricValue(model.throughput_samples_per_sec, 'throughput_samples_per_sec')}</p>
                      <p>Inference: {formatMetricValue(model.inference_time_ms, 'inference_time_ms')}</p>
                    </div>
                    <div className="metric-group">
                      <h4>Resources</h4>
                      <p>Memory: {formatMetricValue(model.memory_usage_mb, 'memory_usage_mb')}</p>
                      <p>Parameters: {formatMetricValue(model.parameters_count, 'parameters_count')}</p>
                      <p>Training Time: {formatMetricValue(model.training_time_hours, 'training_time_hours')}</p>
                      {model.gpu_memory_mb && <p>GPU Memory: {formatMetricValue(model.gpu_memory_mb, 'gpu_memory_mb')}</p>}
                      {model.cpu_usage_percent && <p>CPU Usage: {formatMetricValue(model.cpu_usage_percent, 'cpu_usage_percent')}</p>}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="comparison-insights">
            <h2>Key Insights</h2>
            <div className="insights-grid">
              <div className="insight-card">
                <h3>🚀 Best Latency</h3>
                <p><strong>{getBestModel('latency_ms')}</strong></p>
                <p>Fastest response time for real-time applications</p>
              </div>
              <div className="insight-card">
                <h3>🎯 Best Accuracy</h3>
                <p><strong>{getBestModel('accuracy')}</strong></p>
                <p>Highest overall prediction accuracy</p>
              </div>
              <div className="insight-card">
                <h3>💾 Most Memory Efficient</h3>
                <p><strong>{getBestModel('memory_usage_mb')}</strong></p>
                <p>Lowest memory footprint for deployment</p>
              </div>
              <div className="insight-card">
                <h3>⚡ Best Throughput</h3>
                <p><strong>{getBestModel('throughput_samples_per_sec')}</strong></p>
                <p>Highest processing capacity</p>
              </div>
            </div>
          </div>

          <div className="recommendations">
            <h2>Model Selection Rationale</h2>
            <div className="recommendation-cards">
              <div className="recommendation-card">
                <h3>Why Not Simple RNN?</h3>
                <p>While <strong>Simple RNN</strong> has the lowest latency, it suffers from vanishing gradient problems and achieves low accuracy. Not suitable for complex medical NER tasks requiring long-term dependencies.</p>
              </div>
              <div className="recommendation-card">
                <h3>LSTM vs GRU Trade-offs</h3>
                <p><strong>LSTM</strong> and <strong>GRU</strong> show improvements over RNN but still lag behind BiLSTM. GRU is slightly faster but both lack the bidirectional context crucial for medical entity recognition.</p>
              </div>
              <div className="recommendation-card">
                <h3>Why BiLSTM-CRF?</h3>
                <p><strong>BiLSTM-CRF</strong> achieves higher accuracy with reasonable latency. The bidirectional architecture captures both past and future context, essential for accurate medical NER.</p>
              </div>
              <div className="recommendation-card">
                <h3>Why Transformer (BioBERT)?</h3>
                <p><strong>BioBERT-Transformer</strong> delivers the highest accuracy through pre-training on biomedical literature. Despite higher latency, it's optimal for clinical decision support where accuracy is paramount.</p>
              </div>
              <div className="recommendation-card">
                <h3>Final Recommendation</h3>
                <p>Use <strong>BiLSTM-CRF</strong> for real-time applications requiring speed and <strong>BioBERT-Transformer</strong> for clinical workflows where accuracy is critical. Both significantly outperform simpler architectures.</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}