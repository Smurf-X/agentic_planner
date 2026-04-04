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
      const result = await optimize(yamlText, objective, modelConfigPath);
      onResult(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Optimize failed');
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '16px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <div style={{ marginBottom: '12px' }}>
        <button onClick={() => setMode('generate')} style={{ marginRight: '8px', fontWeight: mode === 'generate' ? 'bold' : 'normal' }}>
          Generate
        </button>
        <button onClick={() => setMode('optimize')} style={{ fontWeight: mode === 'optimize' ? 'bold' : 'normal' }}>
          Optimize
        </button>
      </div>

      <div style={{ marginBottom: '12px' }}>
        <label style={{ display: 'block', marginBottom: '4px' }}>Model Config Path:</label>
        <input
          type="text"
          value={modelConfigPath}
          onChange={(e) => setModelConfigPath(e.target.value)}
          style={{ width: '100%', padding: '8px' }}
          placeholder="e.g., /path/to/models.yaml"
        />
      </div>

      {mode === 'generate' ? (
        <>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px' }}>Intent:</label>
            <textarea
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              style={{ width: '100%', padding: '8px', minHeight: '60px' }}
              placeholder="Describe what you want to do..."
            />
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px' }}>Dataset Path:</label>
            <input
              type="text"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              style={{ width: '100%', padding: '8px' }}
              placeholder="e.g., /path/to/data.jsonl"
            />
          </div>
          <button onClick={handleGenerate} disabled={loading || !intent || !datasetPath || !modelConfigPath}>
            {loading ? 'Generating...' : 'Generate YAML'}
          </button>
        </>
      ) : (
        <>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px' }}>YAML Content or Path:</label>
            <textarea
              value={yamlText}
              onChange={(e) => setYamlText(e.target.value)}
              style={{ width: '100%', padding: '8px', minHeight: '100px' }}
              placeholder="Paste YAML or provide file path..."
            />
          </div>
          <div style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px' }}>Objective:</label>
            <select value={objective} onChange={(e) => setObjective(e.target.value)} style={{ padding: '8px' }}>
              <option value="quality">Quality</option>
              <option value="cost">Cost</option>
              <option value="balanced">Balanced</option>
            </select>
          </div>
          <button onClick={handleOptimize} disabled={loading || !yamlText || !modelConfigPath}>
            {loading ? 'Optimizing...' : 'Optimize YAML'}
          </button>
        </>
      )}

      {error && <div style={{ color: 'red', marginTop: '12px' }}>{error}</div>}
    </div>
  );
}