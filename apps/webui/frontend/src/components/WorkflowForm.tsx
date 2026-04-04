import { useState } from 'react';
import { generate, optimize } from '../api/client';

interface WorkflowFormProps {
  onResult: (result: unknown) => void;
}

export function WorkflowForm({ onResult }: WorkflowFormProps) {
  const [mode, setMode] = useState<'generate' | 'optimize'>('generate');
  const [intent, setIntent] = useState('');
  const [datasetPath, setDatasetPath] = useState('');
  const [modelConfigPath, setModelConfigPath] = useState('');
  const [yamlText, setYamlText] = useState('');
  const [objective, setObjective] = useState('quality');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxIterations, setMaxIterations] = useState(3);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await generate(intent, datasetPath, modelConfigPath);
      onResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generate failed');
    }
    setLoading(false);
  };

  const handleOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await optimize(yamlText, objective, modelConfigPath, {
        max_iterations: maxIterations,
      });
      onResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Optimize failed');
    }
    setLoading(false);
  };

  return (
    <div className="form-section">
      <div className="form-header">
        <h3 className="form-title">Pipeline Configuration</h3>
        <p className="form-subtitle">Generate or optimize your data pipeline</p>
      </div>

      <div className="mode-toggle">
        <button 
          className={`mode-btn ${mode === 'generate' ? 'active' : ''}`}
          onClick={() => setMode('generate')}
        >
          Generate
        </button>
        <button 
          className={`mode-btn ${mode === 'optimize' ? 'active' : ''}`}
          onClick={() => setMode('optimize')}
        >
          Optimize
        </button>
      </div>

      <div className="form-group">
        <label className="form-label">
          Model Config Path <span style={{ color: '#EF4444' }}>*</span>
        </label>
        <input
          type="text"
          className="form-input"
          value={modelConfigPath}
          onChange={(e) => setModelConfigPath(e.target.value)}
          placeholder="/path/to/models.yaml (required for LLM calls)"
        />
        <p style={{ fontSize: '12px', color: '#6B7280', marginTop: '4px' }}>
          YAML file containing LLM model configurations (api_key, base_url, model name)
        </p>
      </div>

      {mode === 'generate' ? (
        <>
          <div className="form-group">
            <label className="form-label">
              Intent <span style={{ color: '#EF4444' }}>*</span>
            </label>
            <textarea
              className="form-textarea"
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              placeholder="Describe your data processing pipeline in natural language, e.g., 'Clean text data and remove duplicates'"
              rows={4}
            />
          </div>
          <div className="form-group">
            <label className="form-label">
              Dataset Path <span style={{ color: '#EF4444' }}>*</span>
            </label>
            <input
              type="text"
              className="form-input"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              placeholder="/path/to/data.jsonl"
            />
          </div>
          <button 
            className="btn btn-primary submit-btn"
            onClick={handleGenerate} 
            disabled={loading || !intent || !datasetPath || !modelConfigPath}
          >
            {loading ? (
              <span className="loading">
                <span className="spinner"></span>
                Generating...
              </span>
            ) : (
              'Generate Pipeline'
            )}
          </button>
        </>
      ) : (
        <>
          <div className="form-group">
            <label className="form-label">
              YAML Configuration <span style={{ color: '#EF4444' }}>*</span>
            </label>
            <textarea
              className="form-textarea"
              value={yamlText}
              onChange={(e) => setYamlText(e.target.value)}
              placeholder="Paste your YAML configuration or provide a file path..."
              rows={6}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Optimization Objective</label>
            <select 
              className="form-select" 
              value={objective} 
              onChange={(e) => setObjective(e.target.value)}
            >
              <option value="quality">Quality - Maximize output quality</option>
              <option value="cost">Cost - Minimize resource usage</option>
              <option value="balanced">Balanced - Trade-off optimization</option>
            </select>
          </div>

          <div className="form-group">
            <button 
              className="btn btn-secondary"
              onClick={() => setShowAdvanced(!showAdvanced)}
              style={{ width: '100%', marginBottom: showAdvanced ? '12px' : '0' }}
            >
              {showAdvanced ? '▼ Hide Advanced' : '▶ Show Advanced'}
            </button>
          </div>

          {showAdvanced && (
            <div className="form-group">
              <label className="form-label">Max Iterations</label>
              <input
                type="number"
                className="form-input"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value) || 3)}
                min={1}
                max={20}
              />
              <p style={{ fontSize: '12px', color: '#6B7280', marginTop: '4px' }}>
                Number of MCTS search iterations (higher = more thorough but slower)
              </p>
            </div>
          )}

          <button 
            className="btn btn-primary submit-btn"
            onClick={handleOptimize} 
            disabled={loading || !yamlText || !modelConfigPath}
          >
            {loading ? (
              <span className="loading">
                <span className="spinner"></span>
                Optimizing...
              </span>
            ) : (
              'Optimize Pipeline'
            )}
          </button>
        </>
      )}

      {error && (
        <div className="result-error" style={{ marginTop: '16px' }}>
          <div className="result-error-title">Error</div>
          <div className="result-error-text">{error}</div>
        </div>
      )}
    </div>
  );
}