import React, { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'

interface User {
  id: string
  email: string
  role: 'general_user' | 'qualified_practitioner'
  credentials?: {
    license_number: string
    specialization: string
    verification_status: boolean
    expiry_date: string
    issuing_authority: string
  }
  is_active: boolean
  created_at: string
  last_login?: string
}

interface AuthProps {
  prefillEmail?: string
  isReauth?: boolean
}

const Auth: React.FC<AuthProps> = ({ prefillEmail, isReauth = false }) => {
  const { login, clearReauthFlag } = useAuth()
  const [isLogin, setIsLogin] = useState(true)
  const [userType, setUserType] = useState<'general_user' | 'qualified_practitioner'>('general_user')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // Form states - pre-fill email if provided
  const [email, setEmail] = useState(prefillEmail || '')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  
  // Practitioner credentials
  const [licenseNumber, setLicenseNumber] = useState('')
  const [specialization, setSpecialization] = useState('')
  const [issuingAuthority, setIssuingAuthority] = useState('')
  const [expiryDate, setExpiryDate] = useState('')

  const resetForm = () => {
    setEmail('')
    setPassword('')
    setConfirmPassword('')
    setLicenseNumber('')
    setSpecialization('')
    setIssuingAuthority('')
    setExpiryDate('')
    setError(null)
    setSuccess(null)
  }

  const handleModeSwitch = (loginMode: boolean) => {
    setIsLogin(loginMode)
    resetForm()
  }

  const validateForm = () => {
    if (!email || !password) {
      setError('Email and password are required')
      return false
    }

    if (!isLogin && password !== confirmPassword) {
      setError('Passwords do not match')
      return false
    }

    if (!isLogin && password.length < 8) {
      setError('Password must be at least 8 characters long')
      return false
    }

    if (!isLogin && password.length > 72) {
      setError('Password cannot be longer than 72 characters')
      return false
    }

    if (!isLogin && userType === 'qualified_practitioner') {
      if (!licenseNumber || !specialization || !issuingAuthority || !expiryDate) {
        setError('All practitioner credentials are required')
        return false
      }

      const expiry = new Date(expiryDate)
      if (expiry <= new Date()) {
        setError('License expiry date must be in the future')
        return false
      }
    }

    return true
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!validateForm()) return

    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      
      if (isLogin) {
        // Login
        const response = await fetch(`${baseUrl}/auth/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email,
            password
          })
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || 'Login failed')
        }

        const tokenData = await response.json()
        
        // Get user info
        const userResponse = await fetch(`${baseUrl}/auth/me`, {
          headers: {
            'Authorization': `Bearer ${tokenData.access_token}`
          }
        })

        if (!userResponse.ok) {
          throw new Error('Failed to get user information')
        }

        const userData = await userResponse.json()
        login(userData, tokenData.access_token)
        
        // Clear re-auth flag if this was a re-authentication
        if (isReauth) {
          clearReauthFlag()
        }
        
      } else {
        // Register
        const requestBody: any = {
          email,
          password,
          role: userType
        }

        if (userType === 'qualified_practitioner') {
          requestBody.credentials = {
            license_number: licenseNumber,
            specialization,
            issuing_authority: issuingAuthority,
            expiry_date: new Date(expiryDate).toISOString()
          }
        }

        const response = await fetch(`${baseUrl}/auth/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody)
        })

        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.detail || 'Registration failed')
        }

        const userData = await response.json()
        
        if (userType === 'qualified_practitioner') {
          setSuccess('Registration successful! Your practitioner credentials are being verified. You can now log in.')
          setIsLogin(true)
          resetForm()
        } else {
          // Auto-login for general users
          const loginResponse = await fetch(`${baseUrl}/auth/login`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              email,
              password
            })
          })

          if (loginResponse.ok) {
            const tokenData = await loginResponse.json()
            login(userData, tokenData.access_token)
          } else {
            setSuccess('Registration successful! Please log in.')
            setIsLogin(true)
            resetForm()
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div 
      className="auth-page"
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '2rem'
      }}
    >
      <div className="auth-container">
        <div className="auth-header">
          <h1>Ayurvedic Clinical Bridge</h1>
          <p>Bridging Traditional Wisdom with Modern Healthcare</p>
        </div>

        <div className="auth-card">
          <div className="auth-tabs">
            <button 
              className={isLogin ? 'active' : ''}
              onClick={() => handleModeSwitch(true)}
            >
              Sign In
            </button>
            <button 
              className={!isLogin ? 'active' : ''}
              onClick={() => handleModeSwitch(false)}
            >
              Sign Up
            </button>
          </div>

          {error && (
            <div className="error-message">
              <p>{error}</p>
            </div>
          )}

          {success && (
            <div className="success-message">
              <p>{success}</p>
            </div>
          )}

          <form onSubmit={handleSubmit} className="auth-form">
            {!isLogin && (
              <div className="user-type-selector">
                <h3>Account Type</h3>
                <div className="radio-group">
                  <label className="radio-option">
                    <input
                      type="radio"
                      name="userType"
                      value="general_user"
                      checked={userType === 'general_user'}
                      onChange={(e) => setUserType(e.target.value as 'general_user')}
                    />
                    <div className="radio-content">
                      <strong>General User</strong>
                      <p>Access basic Ayurvedic information and general wellness guidance</p>
                    </div>
                  </label>
                  
                  <label className="radio-option">
                    <input
                      type="radio"
                      name="userType"
                      value="qualified_practitioner"
                      checked={userType === 'qualified_practitioner'}
                      onChange={(e) => setUserType(e.target.value as 'qualified_practitioner')}
                    />
                    <div className="radio-content">
                      <strong>Healthcare Practitioner</strong>
                      <p>Access clinical tools, detailed analysis, and practitioner resources</p>
                    </div>
                  </label>
                </div>
              </div>
            )}

            <div className="form-group">
              <label htmlFor="email">Email Address</label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="Enter your email address"
                required
                disabled={loading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder={isLogin ? "Enter your password" : "Create a password (8-72 characters)"}
                required
                disabled={loading}
                maxLength={72}
              />
            </div>

            {!isLogin && (
              <div className="form-group">
                <label htmlFor="confirmPassword">Confirm Password</label>
                <input
                  id="confirmPassword"
                  type="password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  placeholder="Confirm your password"
                  required
                  disabled={loading}
                  maxLength={72}
                />
              </div>
            )}

            {!isLogin && userType === 'qualified_practitioner' && (
              <div className="practitioner-credentials">
                <h3>Professional Credentials</h3>
                <div className="credentials-notice">
                  <p>
                    <strong>Verification Required:</strong> All practitioner credentials will be verified 
                    before full access is granted. Please ensure all information is accurate.
                  </p>
                </div>

                <div className="form-group">
                  <label htmlFor="licenseNumber">Professional License Number</label>
                  <input
                    id="licenseNumber"
                    type="text"
                    value={licenseNumber}
                    onChange={(e) => setLicenseNumber(e.target.value)}
                    placeholder="e.g., MD123456, BAMS789012"
                    required
                    disabled={loading}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="specialization">Medical Specialization</label>
                  <select
                    id="specialization"
                    value={specialization}
                    onChange={(e) => setSpecialization(e.target.value)}
                    required
                    disabled={loading}
                  >
                    <option value="">Select your specialization</option>
                    <option value="Ayurveda">Ayurveda (BAMS)</option>
                    <option value="Internal Medicine">Internal Medicine</option>
                    <option value="Family Medicine">Family Medicine</option>
                    <option value="Integrative Medicine">Integrative Medicine</option>
                    <option value="Naturopathy">Naturopathy</option>
                    <option value="Pharmacology">Pharmacology</option>
                    <option value="Herbalism">Clinical Herbalism</option>
                    <option value="Other">Other</option>
                  </select>
                </div>

                <div className="form-group">
                  <label htmlFor="issuingAuthority">Issuing Authority</label>
                  <input
                    id="issuingAuthority"
                    type="text"
                    value={issuingAuthority}
                    onChange={(e) => setIssuingAuthority(e.target.value)}
                    placeholder="e.g., State Medical Board, CCIM, University"
                    required
                    disabled={loading}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="expiryDate">License Expiry Date</label>
                  <input
                    id="expiryDate"
                    type="date"
                    value={expiryDate}
                    onChange={(e) => setExpiryDate(e.target.value)}
                    required
                    disabled={loading}
                    min={new Date().toISOString().split('T')[0]}
                  />
                </div>
              </div>
            )}

            <button type="submit" disabled={loading} className="auth-submit">
              {loading ? (
                <>
                  <span className="spinner">⟳</span>
                  {isLogin ? 'Signing In...' : 'Creating Account...'}
                </>
              ) : (
                isLogin ? 'Sign In' : 'Create Account'
              )}
            </button>
          </form>

          <div className="auth-footer">
            {isLogin ? (
              <p>
                Don't have an account?{' '}
                <button 
                  type="button" 
                  className="link-button"
                  onClick={() => handleModeSwitch(false)}
                >
                  Sign up here
                </button>
              </p>
            ) : (
              <p>
                Already have an account?{' '}
                <button 
                  type="button" 
                  className="link-button"
                  onClick={() => handleModeSwitch(true)}
                >
                  Sign in here
                </button>
              </p>
            )}
          </div>
        </div>

        <div className="auth-info">
          <div className="info-section">
            <h3>For General Users</h3>
            <ul>
              <li>Access basic Ayurvedic knowledge and remedies</li>
              <li>Get general wellness recommendations</li>
              <li>Learn about traditional practices</li>
              <li>No professional verification required</li>
            </ul>
          </div>

          <div className="info-section">
            <h3>For Healthcare Practitioners</h3>
            <ul>
              <li>Access detailed clinical analysis tools</li>
              <li>Herb-drug interaction checking</li>
              <li>Prescription analysis and mapping</li>
              <li>Professional-grade safety assessments</li>
              <li><strong>Requires credential verification</strong></li>
            </ul>
          </div>

          <div className="security-notice">
            <h4>🔒 Security & Privacy</h4>
            <p>
              Your data is encrypted and secure. Professional credentials are verified 
              through appropriate licensing authorities. We comply with healthcare 
              data protection standards.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Auth