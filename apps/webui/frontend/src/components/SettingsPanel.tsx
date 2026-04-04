import { useState } from 'react';
import { testLLM, ModelConfig } from '../api/client';

interface SettingsPanelProps {
  modelConfig: ModelConfig;
  llmTested: boolean;
  onModelConfigChange: (config: ModelConfig) => void;
  onLLMTestedChange: (tested: boolean) => void;
}

export function SettingsPanel({
  modelConfig,
  llmTested,
  onModelConfigChange,
  onLLMTestedChange,
}: SettingsPanelProps) {
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
          message: `Connection successful! Model: ${result.data.model}`,
        });
      } else {
        onLLMTestedChange(false);
        setLLMTestResult({
          ok: false,
          message: result.error || 'Connection failed',
        });
      }
    } catch (e) {
      onLLMTestedChange(false);
      setLLMTestResult({
        ok: false,
        message: e instanceof Error ? e.message : 'Test failed',
      });
    }
    setLLMTesting(false);
  };

  return (
    <div className="form-section">
      <div className="form-header">
        <h3 className="form-title">Settings</h3>
        <p className="form-subtitle">Configure your LLM and application settings</p>
      </div>

      <div
        style={{
          background: '#F0F9FF',
          border: '1px solid #BAE6FD',
          borderRadius: '10px',
          padding: '16px',
          marginBottom: '20px',
        }}
      >
        <h4
          style={{
            fontSize: '14px',
            fontWeight: '600',
            marginBottom: '12px',
            color: '#0369A1',
          }}
        >
          LLM Configuration
        </h4>

        {llmTested && modelConfig.model && (
          <div
            style={{
              marginBottom: '12px',
              padding: '8px 12px',
              borderRadius: '6px',
              fontSize: '13px',
              background: '#D1FAE5',
              color: '#065F46',
              border: '1px solid #A7F3D0',
            }}
          >
            <span style={{ fontWeight: '500' }}>LLM: Configured</span>
            <span style={{ marginLeft: '8px' }}>{modelConfig.model}</span>
            <span style={{ marginLeft: '4px' }}>✓</span>
          </div>
        )}

        <div className="form-group" style={{ marginBottom: '12px' }}>
          <label className="form-label">Base URL</label>
          <input
            type="text"
            className="form-input"
            value={modelConfig.base_url}
            onChange={(e) => {
              onModelConfigChange({ ...modelConfig, base_url: e.target.value });
            }}
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
            onChange={(e) => {
              onModelConfigChange({ ...modelConfig, api_key: e.target.value });
              onLLMTestedChange(false);
            }}
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
            onChange={(e) => {
              onModelConfigChange({ ...modelConfig, model: e.target.value });
              onLLMTestedChange(false);
            }}
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
          <div
            style={{
              marginTop: '12px',
              padding: '10px 12px',
              borderRadius: '6px',
              fontSize: '13px',
              background: llmTestResult.ok ? '#D1FAE5' : '#FEF2F2',
              color: llmTestResult.ok ? '#065F46' : '#991B1B',
              border: `1px solid ${llmTestResult.ok ? '#A7F3D0' : '#FECACA'}`,
            }}
          >
            {llmTestResult.ok ? '✓' : '✗'} {llmTestResult.message}
          </div>
        )}
      </div>
    </div>
  );
}