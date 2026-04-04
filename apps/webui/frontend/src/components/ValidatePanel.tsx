import { useState } from 'react';
import { validate } from '../api/client';

interface ValidatePanelProps {}

export function ValidatePanel({}: ValidatePanelProps) {
  const [yamlText, setYamlText] = useState('');
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleValidate = async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await validate(yamlText);
      setResult(resp as Record<string, unknown>);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Validate failed');
    }
    setLoading(false);
  };

  return (
    <div className="validate-section">
      <div className="form-header">
        <h3 className="form-title">YAML Validator</h3>
        <p className="form-subtitle">Validate your pipeline configuration</p>
      </div>

      <div className="validate-editor">
        <div className="form-group">
          <label className="form-label">YAML Configuration</label>
          <textarea
            className="form-textarea validate-textarea"
            value={yamlText}
            onChange={(e) => setYamlText(e.target.value)}
            placeholder="Paste your YAML configuration here..."
            rows={12}
          />
        </div>

        <div className="validate-actions">
          <button 
            className="btn btn-primary"
            onClick={handleValidate} 
            disabled={loading || !yamlText}
          >
            {loading ? (
              <span className="loading">
                <span className="spinner"></span>
                Validating...
              </span>
            ) : (
              'Validate Configuration'
            )}
          </button>
          <button 
            className="btn btn-secondary"
            onClick={() => { setYamlText(''); setResult(null); setError(null); }}
          >
            Clear
          </button>
        </div>
      </div>

      {error && (
        <div className="result-error" style={{ marginTop: '20px' }}>
          <div className="result-error-title">Error</div>
          <div className="result-error-text">{error}</div>
        </div>
      )}

      {result && (
        <div 
          className={`validate-result ${result.ok ? 'success' : 'error'}`} 
          style={{ marginTop: '20px' }}
        >
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}