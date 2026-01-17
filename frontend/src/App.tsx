import React, { useState } from 'react'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import Auth from './pages/Auth'
import GeneralUsers from './pages/GeneralUsers'
import Clinicians from './pages/Clinicians'
import ModelComparison from './pages/ModelComparison'
import MedicineMapping from './pages/MedicineMapping'
import './styles.css'

type Page = 'general' | 'clinicians' | 'comparison' | 'medicine-mapping'

function AppContent() {
  const { user, isAuthenticated, isLoading, needsReauth, logout } = useAuth()
  const [currentPage, setCurrentPage] = useState<Page>('general')

  if (isLoading) {
    return (
      <div className="app">
        <div className="loading">
          <p>Loading...</p>
        </div>
      </div>
    )
  }

  // Show re-authentication prompt if session expired
  if (needsReauth && user) {
    return (
      <div className="app">
        <div className="reauth-prompt">
          <div>
            <h2>Session Expired</h2>
            <p>Your session has expired. Please log in again to continue.</p>
            <p><strong>Account:</strong> {user.email}</p>
            <Auth prefillEmail={user.email} isReauth={true} />
          </div>
        </div>
      </div>
    )
  }

  if (!isAuthenticated) {
    return <Auth />
  }

  const canAccessClinicians = user?.role === 'qualified_practitioner' && 
    user?.credentials?.verification_status

  const handlePageChange = (page: Page) => {
    if (page === 'clinicians' && !canAccessClinicians) {
      alert('Access to the Clinicians portal requires verified practitioner credentials.')
      return
    }
    setCurrentPage(page)
  }

  const renderPage = () => {
    switch (currentPage) {
      case 'general':
        return <GeneralUsers />
      case 'clinicians':
        return canAccessClinicians ? <Clinicians /> : <GeneralUsers />
      case 'comparison':
        return <ModelComparison />
      case 'medicine-mapping':
        return <MedicineMapping />
      default:
        return <GeneralUsers />
    }
  }

  const getWelcomeMessage = () => {
    if (user?.role === 'qualified_practitioner') {
      if (user.credentials?.verification_status) {
        return `Welcome, Dr. ${user.email.split('@')[0]} (${user.credentials.specialization})`
      } else {
        return `Welcome, ${user.email.split('@')[0]} (Credentials Pending Verification)`
      }
    }
    return `Welcome, ${user?.email.split('@')[0]}`
  }

  return (
    <div className="app">
      <nav className="main-nav">
        <div className="nav-brand">
          <h1>Ayurvedic Clinical Bridge</h1>
        </div>
        <div className="nav-center">
          <div className="user-welcome">
            {getWelcomeMessage()}
          </div>
        </div>
        <div className="nav-links">
          <button 
            className={currentPage === 'general' ? 'active' : ''}
            onClick={() => handlePageChange('general')}
          >
            General Users
          </button>
          <button 
            className={currentPage === 'clinicians' ? 'active' : ''}
            onClick={() => handlePageChange('clinicians')}
            disabled={!canAccessClinicians}
            title={!canAccessClinicians ? 'Requires verified practitioner credentials' : ''}
          >
            Clinicians/Practitioners
            {user?.role === 'qualified_practitioner' && !user?.credentials?.verification_status && (
              <span className="verification-badge">Pending</span>
            )}
          </button>
          <button 
            className={currentPage === 'comparison' ? 'active' : ''}
            onClick={() => handlePageChange('comparison')}
          >
            Model Comparison
          </button>
          <button 
            className={currentPage === 'medicine-mapping' ? 'active' : ''}
            onClick={() => handlePageChange('medicine-mapping')}
          >
            Medicine Mapping
          </button>
          <button 
            className="logout-button"
            onClick={logout}
          >
            Sign Out
          </button>
        </div>
      </nav>
      
      {user?.role === 'qualified_practitioner' && !user?.credentials?.verification_status && (
        <div className="verification-notice">
          <p>
            <strong>Credential Verification Pending:</strong> Your practitioner credentials are being verified. 
            You currently have access to general user features. Full practitioner access will be granted upon verification.
          </p>
        </div>
      )}
      
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  )
}
