import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'

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

interface AuthContextType {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  needsReauth: boolean
  login: (user: User, token: string) => void
  logout: () => void
  updateUser: (user: User) => void
  clearReauthFlag: () => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null)
  const [token, setToken] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [needsReauth, setNeedsReauth] = useState(false)

  useEffect(() => {
    // Check for stored auth data on app load
    const storedToken = localStorage.getItem('auth_token')
    const storedUser = localStorage.getItem('auth_user')

    if (storedToken && storedUser) {
      try {
        const parsedUser = JSON.parse(storedUser)
        setToken(storedToken)
        setUser(parsedUser)
        
        // Verify token is still valid
        verifyToken(storedToken, parsedUser)
      } catch (error) {
        console.error('Error parsing stored user data:', error)
        logout()
      }
    }
    
    setIsLoading(false)
  }, [])

  const verifyToken = async (authToken: string, userData: User) => {
    try {
      const baseUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
      const response = await fetch(`${baseUrl}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${authToken}`
        }
      })

      if (!response.ok) {
        if (response.status === 401) {
          // Token expired - keep user data but mark as needing re-auth
          console.log('Token expired, user needs to re-authenticate')
          setToken(null)
          setNeedsReauth(true)
          localStorage.removeItem('auth_token')
          // Keep user data in localStorage for re-authentication
          return
        }
        throw new Error('Token verification failed')
      }

      const currentUser = await response.json()
      
      // Update user data if it has changed
      if (JSON.stringify(currentUser) !== JSON.stringify(userData)) {
        setUser(currentUser)
        localStorage.setItem('auth_user', JSON.stringify(currentUser))
      }
    } catch (error) {
      console.error('Token verification failed:', error)
      // Only logout completely on non-401 errors
      logout()
    }
  }

  const login = (userData: User, authToken: string) => {
    setUser(userData)
    setToken(authToken)
    setNeedsReauth(false)
    localStorage.setItem('auth_token', authToken)
    localStorage.setItem('auth_user', JSON.stringify(userData))
  }

  const logout = () => {
    setUser(null)
    setToken(null)
    setNeedsReauth(false)
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
  }

  const clearReauthFlag = () => {
    setNeedsReauth(false)
  }

  const updateUser = (userData: User) => {
    setUser(userData)
    localStorage.setItem('auth_user', JSON.stringify(userData))
  }

  const value: AuthContextType = {
    user,
    token,
    isAuthenticated: !!user && !!token,
    isLoading,
    needsReauth,
    login,
    logout,
    updateUser,
    clearReauthFlag
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

// Higher-order component for protecting routes
export function withAuth<P extends object>(
  Component: React.ComponentType<P>,
  requiredRole?: 'general_user' | 'qualified_practitioner'
) {
  return function AuthenticatedComponent(props: P) {
    const { user, isAuthenticated, isLoading } = useAuth()

    if (isLoading) {
      return (
        <div className="loading">
          <p>Loading...</p>
        </div>
      )
    }

    if (!isAuthenticated) {
      return (
        <div className="auth-required">
          <h2>Authentication Required</h2>
          <p>Please log in to access this page.</p>
        </div>
      )
    }

    if (requiredRole && user?.role !== requiredRole) {
      return (
        <div className="access-denied">
          <h2>Access Denied</h2>
          <p>
            {requiredRole === 'qualified_practitioner' 
              ? 'This page is only accessible to verified healthcare practitioners.'
              : 'You do not have permission to access this page.'
            }
          </p>
        </div>
      )
    }

    return <Component {...props} />
  }
}

// Hook for making authenticated API requests
export function useAuthenticatedFetch() {
  const { token, logout } = useAuth()

  const authenticatedFetch = async (url: string, options: RequestInit = {}) => {
    if (!token) {
      throw new Error('No authentication token available')
    }

    const headers = {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
      ...options.headers
    }

    const response = await fetch(url, {
      ...options,
      headers
    })

    if (response.status === 401) {
      // Token expired or invalid
      logout()
      throw new Error('Authentication expired. Please log in again.')
    }

    return response
  }

  return authenticatedFetch
}