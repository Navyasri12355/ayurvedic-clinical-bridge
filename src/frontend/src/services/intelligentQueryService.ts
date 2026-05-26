/**
 * Intelligent Query Service
 * 
 * This service handles communication with the new intelligent routing API
 * that automatically selects between BiLSTM-CRF, BioBERT, and Knowledge Base
 * based on user type and query complexity.
 */

export interface IntelligentQueryRequest {
  query: string
  preferred_accuracy?: 'balanced' | 'high'
  context?: Record<string, any>
  include_routing_info?: boolean
  include_explanation?: boolean
}

export interface BatchQueryRequest {
  queries: string[]
  preferred_accuracy?: 'balanced' | 'high'
  include_routing_info?: boolean
}

export interface EntityResponse {
  type: string
  text: string
  start?: number
  end?: number
  confidence: number
}

export interface KnowledgeResultResponse {
  id: string
  title: string
  content: Record<string, any>
  knowledge_type: string
  evidence_level: string
  confidence: number
  source: string
}

export interface InteractionResponse {
  herb: string
  drug: string
  severity: string
  mechanism: string
  confidence: number
  evidence_level: string
}

export interface SafetyAssessmentResponse {
  overall_risk_score: number
  risk_level: string
  risk_factors: Record<string, number>
  recommendations: string[]
  requires_consultation: boolean
}

export interface TreatmentRecommendationResponse {
  condition: string
  herb: string
  dosage: string
  formulation: string
  duration: string
  mechanism: string
  evidence_level: string
  confidence: number
}

export interface RoutingDecisionResponse {
  model_used: string
  confidence: number
  reasoning: string
  query_complexity: string
  query_category: string
  estimated_processing_time: number
}

export interface ExplanationResponse {
  feature_importance: Record<string, number>
  explanation_text: string
  confidence_breakdown: Record<string, number>
  model_reasoning: string
}

export interface IntelligentQueryResponse {
  query: string
  model_used: string
  user_role: string
  processing_time: number
  entities: EntityResponse[]
  knowledge_results: KnowledgeResultResponse[]
  interactions: InteractionResponse[]
  safety_assessment?: SafetyAssessmentResponse
  treatment_recommendations: TreatmentRecommendationResponse[]
  confidence_scores: Record<string, number>
  routing_decision?: RoutingDecisionResponse
  explanation?: ExplanationResponse
  metadata: Record<string, any>
}

export interface BatchQueryResponse {
  results: IntelligentQueryResponse[]
  total_queries: number
  total_processing_time: number
  average_processing_time: number
}

export interface ModelCapabilitiesResponse {
  available_models: string[]
  model_descriptions: Record<string, string>
  routing_criteria: Record<string, any>
  supported_query_types: string[]
}

export class IntelligentQueryService {
  private baseUrl: string
  private authToken?: string

  constructor(baseUrl?: string) {
    this.baseUrl = baseUrl || import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'
  }

  setAuthToken(token: string) {
    this.authToken = token
  }

  private async makeRequest<T>(
    endpoint: string, 
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`
    
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...options.headers as Record<string, string>
    }

    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`
    }

    const response = await fetch(url, {
      ...options,
      headers
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    return response.json()
  }

  /**
   * Process a single query using intelligent routing
   */
  async processQuery(request: IntelligentQueryRequest): Promise<IntelligentQueryResponse> {
    return this.makeRequest<IntelligentQueryResponse>('/api/intelligent-query/process', {
      method: 'POST',
      body: JSON.stringify(request)
    })
  }

  /**
   * Compare BiLSTM-CRF vs BioBERT models for sequence tagging analysis
   */
  async compareModels(request: IntelligentQueryRequest): Promise<Record<string, any>> {
    return this.makeRequest<Record<string, any>>('/api/intelligent-query/compare-models', {
      method: 'POST',
      body: JSON.stringify(request)
    })
  }

  /**
   * Comprehensive evaluation of all models
   */
  async evaluateModels(request: BatchQueryRequest): Promise<Record<string, any>> {
    return this.makeRequest<Record<string, any>>('/api/intelligent-query/evaluate-models', {
      method: 'POST',
      body: JSON.stringify(request)
    })
  }

  /**
   * Process multiple queries in batch
   */
  async processBatchQueries(request: BatchQueryRequest): Promise<BatchQueryResponse> {
    return this.makeRequest<BatchQueryResponse>('/api/intelligent-query/batch-process', {
      method: 'POST',
      body: JSON.stringify(request)
    })
  }

  /**
   * Get model capabilities and routing information
   */
  async getCapabilities(): Promise<ModelCapabilitiesResponse> {
    return this.makeRequest<ModelCapabilitiesResponse>('/api/intelligent-query/capabilities')
  }

  /**
   * Get routing statistics and usage information
   */
  async getRoutingStats(): Promise<Record<string, any>> {
    return this.makeRequest<Record<string, any>>('/api/intelligent-query/routing-stats')
  }

  /**
   * Health check for the intelligent query service
   */
  async healthCheck(): Promise<Record<string, any>> {
    return this.makeRequest<Record<string, any>>('/api/intelligent-query/health')
  }
}

// Global service instance
export const intelligentQueryService = new IntelligentQueryService()

// Helper functions for common query patterns
export const queryHelpers = {
  /**
   * Create a general information query (routes to Knowledge Base)
   */
  createGeneralQuery: (query: string): IntelligentQueryRequest => ({
    query,
    preferred_accuracy: 'fast',
    include_routing_info: true
  }),

  /**
   * Create a medicine mapping query (routes to BiLSTM-CRF for general users)
   */
  createMedicineMappingQuery: (query: string): IntelligentQueryRequest => ({
    query,
    preferred_accuracy: 'balanced',
    include_routing_info: true
  }),

  /**
   * Create a clinical analysis query (routes to BioBERT for practitioners)
   */
  createClinicalQuery: (query: string): IntelligentQueryRequest => ({
    query,
    preferred_accuracy: 'high',
    include_routing_info: true
  }),

  /**
   * Create a safety assessment query (routes to BioBERT)
   */
  createSafetyQuery: (herbs: string[], drugs: string[]): IntelligentQueryRequest => ({
    query: `Analyze safety and interactions between herbs: ${herbs.join(', ')} and drugs: ${drugs.join(', ')}`,
    preferred_accuracy: 'high',
    context: { herbs, drugs, query_type: 'safety_assessment' },
    include_routing_info: true
  }),

  /**
   * Create a treatment planning query (routes to BioBERT for practitioners)
   */
  createTreatmentQuery: (condition: string): IntelligentQueryRequest => ({
    query: `Comprehensive Ayurvedic treatment protocol for ${condition}`,
    preferred_accuracy: 'high',
    context: { condition, query_type: 'treatment_planning' },
    include_routing_info: true
  }),

  /**
   * Create a comparative analysis request for BiLSTM-CRF vs BioBERT
   */
  createComparativeQuery: (query: string): IntelligentQueryRequest => ({
    query,
    preferred_accuracy: 'balanced',
    include_routing_info: true,
    context: { analysis_type: 'comparative', models: ['bilstm_crf', 'biobert'] }
  }),

  /**
   * Create batch evaluation request for model comparison
   */
  createEvaluationBatch: (queries: string[]): BatchQueryRequest => ({
    queries,
    preferred_accuracy: 'balanced',
    include_routing_info: true
  })
}

export default IntelligentQueryService