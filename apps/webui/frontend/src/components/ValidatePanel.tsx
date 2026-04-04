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
    <div style={{ padding: '16px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h3 style={{ marginBottom: '12px' }}>Validate YAML</h3>
      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'block', marginBottom: '4px' }}>YAML Content or Path:</label>
        <textarea
          value={yamlText}
          onChange={(e) => setYamlText(e.target.value)}
          style={{ width: '100%', padding: '8px', minHeight: '100px' }}
          placeholder="Paste YAML or provide file path..."
        />
      </div>
      <button onClick={handleValidate} disabled={loading || !yamlText}>
        {loading ? 'Validating...' : 'Validate'}
      </button>
      {error && <div style={{ color: 'red', marginTop: '12px' }}>{error}</div>}
      {result && (
        <div style={{ marginTop: '16px', padding: '12px', background: result.ok ? '#f0fff0' : '#fff0f0', borderRadius: '4px' }}>
          <pre style={{ overflow: 'auto', maxHeight: '150px' }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}