import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

interface MedicineMapping {
  allopathic_medicine: string
  ayurvedic_alternatives: string[]
  disease: string
  dosage: string
  formulation: string
  dosha: string
  constitution: string
  confidence_score: number
  safety_notes: string
  contraindications: string
  interaction_warnings: string[]
}

interface AyurvedicRecommendation {
  herb_name: string
  dosage: string
  formulation: string
  preparation_method: string
  timing: string
  duration: string
  precautions: string
}

export default function MedicineMapping() {
  const { user } = useAuth()
  const [activeTab, setActiveTab] = useState<'mapping' | 'disease' | 'symptoms'>('mapping')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Medicine mapping state
  const [allopathicMedicine, setAllopathicMedicine] = useState('')
  const [disease, setDisease] = useState('')
  const [mappingResult, setMappingResult] = useState<MedicineMapping | null>(null)
  
  // Disease recommendations state
  const [diseaseQuery, setDiseaseQuery] = useState('')
  const [diseaseRecommendations, setDiseaseRecommendations] = useState<AyurvedicRecommendation[]>([])
  
  // Symptoms search state
  const [symptoms, setSymptoms] = useState('')
  const [symptomResults, setSymptomResults] = useState<any[]>([])

  const findAlternative = async () => {
    if (!allopathicMedicine.trim()) {
      setError('Please enter an allopathic medicine name')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await fetch(`${baseUrl}/api/medicine-mapping/find-alternative`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          allopathic_medicine: allopathicMedicine,
          disease: disease || null
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to find alternative')
      }

      const result = await response.json()
      setMappingResult(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const getDiseaseRecommendations = async () => {
    if (!diseaseQuery.trim()) {
      setError('Please enter a disease name')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await fetch(`${baseUrl}/api/medicine-mapping/disease-recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          disease: diseaseQuery
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to get recommendations')
      }

      const result = await response.json()
      setDiseaseRecommendations(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const searchBySymptoms = async () => {
    if (!symptoms.trim()) {
      setError('Please enter symptoms')
      return
    }

    setLoading(true)
    setError(null)
    
    try {
      const symptomList = symptoms.split(',').map(s => s.trim()).filter(s => s)
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await fetch(`${baseUrl}/api/medicine-mapping/search-by-symptoms`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          symptoms: symptomList
        })
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to search by symptoms')
      }

      const result = await response.json()
      setSymptomResults(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="medicine-mapping">
      <div className="page-header">
        <h2>Medicine Mapping</h2>
        <p>Find Ayurvedic alternatives to allopathic medicines</p>
      </div>

      <div className="tabs">
        <button 
          className={activeTab === 'mapping' ? 'active' : ''}
          onClick={() => setActiveTab('mapping')}
        >
          Medicine Mapping
        </button>
        <button 
          className={activeTab === 'disease' ? 'active' : ''}
          onClick={() => setActiveTab('disease')}
        >
          Disease Recommendations
        </button>
        <button 
          className={activeTab === 'symptoms' ? 'active' : ''}
          onClick={() => setActiveTab('symptoms')}
        >
          Symptom Search
        </button>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {activeTab === 'mapping' && (
        <div className="tab-content">
          <div className="input-section">
            <div className="input-group">
              <label>Allopathic Medicine *</label>
              <input
                type="text"
                value={allopathicMedicine}
                onChange={(e) => setAllopathicMedicine(e.target.value)}
                placeholder="e.g., Metformin, Ibuprofen, Omeprazole"
              />
            </div>
            <div className="input-group">
              <label>Disease (Optional)</label>
              <input
                type="text"
                value={disease}
                onChange={(e) => setDisease(e.target.value)}
                placeholder="e.g., Diabetes, Hypertension"
              />
            </div>
            <button 
              onClick={findAlternative}
              disabled={loading}
              className="primary-button"
            >
              {loading ? 'Searching...' : 'Find Ayurvedic Alternative'}
            </button>
          </div>

          {mappingResult && (
            <div className="results-section">
              <h3>Ayurvedic Alternative Found</h3>
              <div className="mapping-result">
                <div className="result-header">
                  <h4>{mappingResult.allopathic_medicine}</h4>
                  <span className="confidence-score">
                    Confidence: {(mappingResult.confidence_score * 100).toFixed(0)}%
                  </span>
                </div>
                
                <div className="result-details">
                  <div className="detail-group">
                    <strong>Ayurvedic Alternatives:</strong>
                    <ul>
                      {mappingResult.ayurvedic_alternatives.map((herb, index) => (
                        <li key={index}>{herb}</li>
                      ))}
                    </ul>
                  </div>
                  
                  <div className="detail-group">
                    <strong>Dosage:</strong> {mappingResult.dosage}
                  </div>
                  
                  <div className="detail-group">
                    <strong>Formulation:</strong> {mappingResult.formulation}
                  </div>
                  
                  <div className="detail-group">
                    <strong>Dosha:</strong> {mappingResult.dosha}
                  </div>
                  
                  {mappingResult.safety_notes && (
                    <div className="detail-group safety-notes">
                      <strong>Safety Notes:</strong> {mappingResult.safety_notes}
                    </div>
                  )}
                  
                  {mappingResult.contraindications && (
                    <div className="detail-group contraindications">
                      <strong>Contraindications:</strong> {mappingResult.contraindications}
                    </div>
                  )}
                  
                  {mappingResult.interaction_warnings.length > 0 && (
                    <div className="detail-group warnings">
                      <strong>Interaction Warnings:</strong>
                      <ul>
                        {mappingResult.interaction_warnings.map((warning, index) => (
                          <li key={index}>{warning}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'disease' && (
        <div className="tab-content">
          <div className="input-section">
            <div className="input-group">
              <label>Disease Name *</label>
              <input
                type="text"
                value={diseaseQuery}
                onChange={(e) => setDiseaseQuery(e.target.value)}
                placeholder="e.g., Diabetes, Arthritis, Migraine"
              />
            </div>
            <button 
              onClick={getDiseaseRecommendations}
              disabled={loading}
              className="primary-button"
            >
              {loading ? 'Searching...' : 'Get Ayurvedic Recommendations'}
            </button>
          </div>

          {diseaseRecommendations.length > 0 && (
            <div className="results-section">
              <h3>Ayurvedic Recommendations for {diseaseQuery}</h3>
              <div className="recommendations-grid">
                {diseaseRecommendations.map((rec, index) => (
                  <div key={index} className="recommendation-card">
                    <h4>{rec.herb_name}</h4>
                    <div className="rec-details">
                      <p><strong>Dosage:</strong> {rec.dosage}</p>
                      <p><strong>Preparation:</strong> {rec.preparation_method}</p>
                      <p><strong>Timing:</strong> {rec.timing}</p>
                      <p><strong>Duration:</strong> {rec.duration}</p>
                      {rec.precautions && (
                        <p><strong>Precautions:</strong> {rec.precautions}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'symptoms' && (
        <div className="tab-content">
          <div className="input-section">
            <div className="input-group">
              <label>Symptoms (comma-separated) *</label>
              <input
                type="text"
                value={symptoms}
                onChange={(e) => setSymptoms(e.target.value)}
                placeholder="e.g., headache, nausea, fatigue"
              />
            </div>
            <button 
              onClick={searchBySymptoms}
              disabled={loading}
              className="primary-button"
            >
              {loading ? 'Searching...' : 'Search by Symptoms'}
            </button>
          </div>

          {symptomResults.length > 0 && (
            <div className="results-section">
              <h3>Possible Conditions and Treatments</h3>
              <div className="symptom-results">
                {symptomResults.map((result, index) => (
                  <div key={index} className="symptom-result-card">
                    <h4>{result.disease}</h4>
                    <p><strong>Symptoms:</strong> {result.symptoms}</p>
                    <p><strong>Ayurvedic Herbs:</strong> {result.ayurvedic_herbs}</p>
                    <p><strong>Dosha:</strong> {result.dosha}</p>
                    {result.diet_recommendations && (
                      <p><strong>Diet Recommendations:</strong> {result.diet_recommendations}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="disclaimer">
        <p><strong>Disclaimer:</strong> This information is for educational purposes only. 
        Always consult with qualified healthcare professionals before making any changes 
        to your medication regimen or starting new treatments.</p>
      </div>
    </div>
  )
}