import React, { useState, useEffect } from 'react'
import { useAuthenticatedFetch } from '../contexts/AuthContext'

interface PendingPractitioner {
  id: string
  email: string
  credentials: {
    license_number: string
    specialization: string
    verification_status: boolean
    expiry_date: string
    issuing_authority: string
  }
  created_at: string
}

export default function AdminVerification() {
  const authenticatedFetch = useAuthenticatedFetch()
  const [pendingPractitioners, setPendingPractitioners] = useState<PendingPractitioner[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [processingId, setProcessingId] = useState<string | null>(null)

  useEffect(() => {
    fetchPendingPractitioners()
  }, [])

  const fetchPendingPractitioners = async () => {
    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await authenticatedFetch(`${baseUrl}/admin/pending-practitioners`)
      
      if (!response.ok) {
        throw new Error('Failed to fetch pending practitioners')
      }
      
      const data = await response.json()
      setPendingPractitioners(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleVerification = async (practitionerId: string, approve: boolean) => {
    setProcessingId(practitionerId)
    setError(null)

    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await authenticatedFetch(`${baseUrl}/admin/verify-practitioner`, {
        method: 'POST',
        body: JSON.stringify({
          practitioner_id: practitionerId,
          approved: approve
        })
      })

      if (!response.ok) {
        throw new Error('Failed to process verification')
      }

      // Remove the practitioner from the pending list
      setPendingPractitioners(prev => 
        prev.filter(p => p.id !== practitionerId)
      )
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setProcessingId(null)
    }
  }

  if (loading) {
    return (
      <div className="admin-verification">
        <div className="loading">
          <p>Loading pending verifications...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="admin-verification">
      <div className="header">
        <h1>Practitioner Credential Verification</h1>
        <p>Review and verify healthcare practitioner credentials</p>
      </div>

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {pendingPractitioners.length === 0 ? (
        <div className="no-pending">
          <h3>No Pending Verifications</h3>
          <p>All practitioner credentials have been processed.</p>
        </div>
      ) : (
        <div className="pending-list">
          <h2>Pending Verifications ({pendingPractitioners.length})</h2>
          
          {pendingPractitioners.map((practitioner) => (
            <div key={practitioner.id} className="verification-card">
              <div className="practitioner-details">
                <h3>{practitioner.email}</h3>
                <div className="credential-info">
                  <div className="info-row">
                    <span className="label">License Number:</span>
                    <span className="value">{practitioner.credentials.license_number}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Specialization:</span>
                    <span className="value">{practitioner.credentials.specialization}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Issuing Authority:</span>
                    <span className="value">{practitioner.credentials.issuing_authority}</span>
                  </div>
                  <div className="info-row">
                    <span className="label">Expiry Date:</span>
                    <span className="value">
                      {new Date(practitioner.credentials.expiry_date).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="info-row">
                    <span className="label">Applied:</span>
                    <span className="value">
                      {new Date(practitioner.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </div>
              
              <div className="verification-actions">
                <button
                  className="approve-button"
                  onClick={() => handleVerification(practitioner.id, true)}
                  disabled={processingId === practitioner.id}
                >
                  {processingId === practitioner.id ? 'Processing...' : 'Approve'}
                </button>
                <button
                  className="reject-button"
                  onClick={() => handleVerification(practitioner.id, false)}
                  disabled={processingId === practitioner.id}
                >
                  {processingId === practitioner.id ? 'Processing...' : 'Reject'}
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}