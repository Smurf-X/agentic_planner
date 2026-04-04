const API_BASE = '/api';

interface ToolResponse {
  ok: boolean;
  data: Record<string, unknown>;
  timing_ms: number;
  error?: string;
  token_usage?: Record<string, unknown>;
}

interface GenerateOptions {
  use_real_generator?: boolean;
  retrieval_mode?: 'none' | 'bm25' | 'vector';
  candidate_top_k?: number;
  dataset_hint?: string;
}

interface OptimizeOptions {
  use_real_optimizer?: boolean;
  max_iterations?: number;
  max_evaluations?: number;
  optimize_mode?: 'search_only' | 'evaluate_only' | 'search_and_evaluate';
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return resp.json();
}

export async function dispatch(action: string, payload: Record<string, unknown>): Promise<ToolResponse> {
  return post<ToolResponse>('/dispatch', { action, payload });
}

export async function generate(
  intent: string, 
  dataset_path: string, 
  model_config_path: string, 
  options?: GenerateOptions
): Promise<ToolResponse> {
  const fullOptions: GenerateOptions = {
    use_real_generator: true,
    retrieval_mode: 'none',
    candidate_top_k: 20,
    ...options,
  };
  return post<ToolResponse>('/generate', { 
    intent, 
    dataset_path, 
    model_config_path, 
    options: fullOptions 
  });
}

export async function optimize(
  yaml_text_or_path: string, 
  objective: string, 
  model_config_path: string, 
  options?: OptimizeOptions
): Promise<ToolResponse> {
  const fullOptions: OptimizeOptions = {
    use_real_optimizer: true,
    max_iterations: 3,
    max_evaluations: 100,
    optimize_mode: 'search_only',
    ...options,
  };
  return post<ToolResponse>('/optimize', { 
    yaml_text_or_path, 
    objective, 
    model_config_path, 
    options: fullOptions 
  });
}

export async function validate(yaml_text_or_path: string, options?: Record<string, unknown>): Promise<ToolResponse> {
  return post<ToolResponse>('/validate', { yaml_text_or_path, options: options || {} });
}

export async function listOps(): Promise<ToolResponse> {
  return post<ToolResponse>('/list_ops', {});
}

export async function explainOp(operator_name: string, options?: Record<string, unknown>): Promise<ToolResponse> {
  return post<ToolResponse>('/explain_op', { operator_name, options: options || {} });
}