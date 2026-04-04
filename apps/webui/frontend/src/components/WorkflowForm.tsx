import { useState } from 'react';
import { generate, optimize, testLLM, ModelConfig } from '../api/client';

interface WorkflowFormProps {
  onResult: (result: unknown) => void;
  modelConfig: ModelConfig;
  llmTested: boolean;
  onModelConfigChange: (config: ModelConfig) => void;
  onLLMTestedChange: (tested: boolean) => void;
}

export function WorkflowForm({ 
  onResult, 
  modelConfig, 
  llmTested, 
  onModelConfigChange, 
  onLLMTestedChange 
}: WorkflowFormProps) {
  const [mode, setMode] = useState<'generate' | 'optimize'>('generate');
  const [intent, setIntent] = useState('');
  const [datasetPath, setDatasetPath] = useState('');
  const [yamlText, setYamlText] = useState('');
  const [objective, setObjective] = useState('quality');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [maxIterations, setMaxIterations] = useState(3);
  const [llmTesting, setLLMTesting] = useState(false);
  const [llmTestResult, setLLMTestResult] = useState<{ ok: boolean; message: string } | null>(null);

  const handleTestLLM = async () => {
    if (!modelConfig.model || !modelConfig.api_key) {
      setLLMTestResult({ ok: false, message: 'Please fill in Model Name and API Key' });
      return;
    }

    setLLMTesting(true);
    setLLMTestResult(null);
    try {
      const result = await testLLM(modelConfig);
      if (result.ok) {
        onLLMTestedChange(true);
        setLLMTestResult({ 
          ok: true, 
          message: `Connection successful! Model: ${result.data.model}` 
        });
      } else {
        onLLMTestedChange(false);
        setLLMTestResult({ 
          ok: false, 
          message: result.error || 'Connection failed' 
        });
      }
    } catch (e) {
      onLLMTestedChange(false);
      setLLMTestResult({ 
        ok: false, 
        message: e instanceof Error ? e.message : 'Test failed' 
      });
    }
    setLLMTesting(false);
  };

  const handleGenerate = async () => {
    if (!llmTested) {
      setError('Please test LLM connection first');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await generate(intent, datasetPath, modelConfig);
      onResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Generate failed');
    }
    setLoading(false);
  };

  const handleOptimize = async () => {
    if (!llmTested) {
      setError('Please test LLM connection first');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await optimize(yamlText, objective, modelConfig, {
        max_iterations: maxIterations,
      });
      onResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Optimize failed');
    }
    setLoading(false);
  };

  const canSubmit = llmTested && (
    (mode === 'generate' && intent && datasetPath) ||
    (mode === 'optimize' && yamlText)
  );

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

      <div style={{ 
        background: '#F0F9FF', 
        border: '1px solid #BAE6FD', 
        borderRadius: '10px', 
        padding: '16px',
        marginBottom: '20px'
      }}>
        <h4 style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#0369A1' }}>
          LLM Configuration
        </h4>
        
        <div className="form-group" style={{ marginBottom: '12px' }}>
          <label className="form-label">Base URL</label>
          <input
            type="text"
            className="form-input"
            value={modelConfig.base_url}
            onChange={(e) => { onModelConfigChange({ ...modelConfig, base_url: e.target.value }); onLLMTestedChange(false); }}
            placeholder="https://api.openai.com/v1 (leave empty for OpenAI)"
          />
        </div>

        <div className="form-group" style={{ marginBottom: '12px' }}>
          <label className="form-label">
            API Key <span style={{ color: '#EF4444' }}>*</span>
          </label>
          <input
            type="password"
            className="form-input"
            value={modelConfig.api_key}
            onChange={(e) => { onModelConfigChange({ ...modelConfig, api_key: e.target.value }); onLLMTestedChange(false); }}
            placeholder="sk-..."
          />
        </div>

        <div className="form-group" style={{ marginBottom: '12px' }}>
          <label className="form-label">
            Model Name <span style={{ color: '#EF4444' }}>*</span>
          </label>
          <input
            type="text"
            className="form-input"
            value={modelConfig.model}
            onChange={(e) => { onModelConfigChange({ ...modelConfig, model: e.target.value }); onLLMTestedChange(false); }}
            placeholder="gpt-4o-mini"
          />
        </div>

        <button 
          className={`btn ${llmTested ? 'btn-secondary' : 'btn-primary'}`}
          onClick={handleTestLLM}
          disabled={llmTesting}
          style={{ width: '100%' }}
        >
          {llmTesting ? (
            <span className="loading">
              <span className="spinner"></span>
              Testing...
            </span>
          ) : llmTested ? (
            'Test Again'
          ) : (
            'Test Connection'
          )}
        </button>

        {llmTestResult && (
          <div style={{ 
            marginTop: '12px',
            padding: '10px 12px',
            borderRadius: '6px',
            fontSize: '13px',
            background: llmTestResult.ok ? '#D1FAE5' : '#FEF2F2',
            color: llmTestResult.ok ? '#065F46' : '#991B1B',
            border: `1px solid ${llmTestResult.ok ? '#A7F3D0' : '#FECACA'}`
          }}>
            {llmTestResult.ok ? '✓' : '✗'} {llmTestResult.message}
          </div>
        )}
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
            disabled={loading || !canSubmit}
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
              placeholder="Paste your YAML configuration..."
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
            </div>
          )}

          <button 
            className="btn btn-primary submit-btn"
            onClick={handleOptimize} 
            disabled={loading || !canSubmit}
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