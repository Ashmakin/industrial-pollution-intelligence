import axios from 'axios';

// API base configuration
const API_BASE_URL = 'http://localhost:8080';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Generic API response type
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// Data types
export interface WaterQualityRecord {
  id: number;
  station_name: string;
  station_code?: string;
  province?: string;
  watershed?: string;
  monitoring_time: string;
  temperature?: number;
  ph?: number;
  dissolved_oxygen?: number;
  conductivity?: number;
  turbidity?: number;
  permanganate_index?: number;
  ammonia_nitrogen?: number;
  total_phosphorus?: number;
  total_nitrogen?: number;
  chlorophyll_a?: number;
  algae_density?: number;
  water_quality_grade?: number;
  pollution_index?: number;
  data_source?: string;
  created_at: string;
  updated_at: string;
}

export interface StationInfo {
  station_name: string;
  station_code?: string;
  province?: string;
  watershed?: string;
  measurement_count: number;
  latest_measurement_time?: string;
}

export interface ParameterStatistics {
  parameter: string;
  count: number;
  mean?: number;
  std_dev?: number;
  min?: number;
  max?: number;
}

export interface ForecastRequest {
  station: string;
  parameter: string;
  horizon: number; // hours
  model: 'lstm' | 'prophet' | 'ensemble';
}

export interface ForecastResult {
  station_name: string;
  parameter: string;
  predictions: Array<{
    timestamp: string;
    predicted_value: number;
    confidence_lower?: number;
    confidence_upper?: number;
  }>;
  model_metrics?: {
    rmse: number;
    mae: number;
    mape: number;
  };
}

export interface DataCollectionRequest {
  area_ids: number[];
  start_date?: string;
  end_date?: string;
  max_records?: number;
}

export interface AnalysisRequest {
  analysis_type: 'pca' | 'granger' | 'anomaly' | 'correlation';
  stations?: string[];
  parameters?: string[];
  time_range?: {
    start: string;
    end: string;
  };
}

// Error handling utility
export const handleApiError = (error: any): string => {
  if (error.response?.data?.message) {
    return error.response.data.message;
  }
  if (error.response?.data?.error) {
    return error.response.data.error;
  }
  if (error.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

// Health check
export const checkHealth = async (): Promise<ApiResponse<any>> => {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

// Water quality data endpoints

export const getMeasurements = async (params?: {
  station_name?: string;
  province?: string;
  watershed?: string;
  parameter?: string;
  page?: number;
  limit?: number;
}): Promise<ApiResponse<WaterQualityRecord[]>> => {
  try {
    const response = await apiClient.get('/api/pollution/measurements', { params });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

export const getStatistics = async (params?: {
  station_name?: string;
  province?: string;
  watershed?: string;
}): Promise<ApiResponse<ParameterStatistics[]>> => {
  try {
    const response = await apiClient.get('/api/pollution/statistics', { params });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

// Data collection endpoints
export const getCollectionStatus = async (): Promise<ApiResponse<any>> => {
  try {
    const response = await apiClient.get('/api/data/status');
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

// Forecasting endpoints
export const generateForecast = async (request: ForecastRequest): Promise<ApiResponse<ForecastResult>> => {
  try {
    const response = await apiClient.post('/api/forecast/generate', request);
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

export const getForecasts = async (params?: {
  station_name?: string;
  parameter?: string;
}): Promise<ApiResponse<ForecastResult[]>> => {
  try {
    const response = await apiClient.get('/api/forecast/list', { params });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

// Analysis endpoints
export const runAnalysis = async (request: AnalysisRequest): Promise<ApiResponse<any>> => {
  try {
    const analysisType = request.analysis_type || 'pca';
    const response = await apiClient.get(`/api/analysis/${analysisType}`, {
      params: {
        analysis_type: analysisType,
        station_name: request.stations?.join(','),
        parameter: request.parameters?.join(',')
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

export const getAnalysisResults = async (params?: {
  analysis_type?: string;
  station_name?: string;
  parameter?: string;
}): Promise<ApiResponse<any[]>> => {
  try {
    const response = await apiClient.get('/api/analysis/results', { params });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

// Reports endpoints
export const generateReport = async (params: {
  report_type: 'summary' | 'detailed' | 'forecast';
  stations?: string[];
  time_range?: {
    start: string;
    end: string;
  };
}): Promise<ApiResponse<any>> => {
  try {
    const response = await apiClient.post('/api/reports/generate', params);
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};

// Enhanced data collection functions
export const startDataCollection = async (request: {
  areas?: string[];
  basins?: string[];
  stations?: string[];
  max_records?: number;
}): Promise<ApiResponse<any>> => {
  try {
    const response = await apiClient.post('/api/data/collect', request);
    return response.data;
  } catch (error: any) {
    console.error('Data collection error:', error);
    return {
      success: false,
      error: error.response?.data?.error || error.message || '数据采集失败'
    };
  }
};

// Get available areas
export const getAreas = async (): Promise<ApiResponse<any[]>> => {
  try {
    const response = await apiClient.get('/api/areas');
    return response.data;
  } catch (error: any) {
    console.error('Get areas error:', error);
    return {
      success: false,
      error: error.response?.data?.error || error.message || '获取地区列表失败'
    };
  }
};

// Get available basins
export const getBasins = async (): Promise<ApiResponse<any[]>> => {
  try {
    const response = await apiClient.get('/api/basins');
    return response.data;
  } catch (error: any) {
    console.error('Get basins error:', error);
    return {
      success: false,
      error: error.response?.data?.error || error.message || '获取流域列表失败'
    };
  }
};

export const downloadReport = async (reportId: string): Promise<Blob> => {
  try {
    const response = await apiClient.get(`/api/reports/download/${reportId}`, {
      responseType: 'blob'
    });
    return response.data;
  } catch (error) {
    throw new Error(handleApiError(error));
  }
};