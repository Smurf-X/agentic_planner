interface ResultPanelProps {
  result: unknown | null;
}

export function ResultPanel({ result }: ResultPanelProps) {
  if (!result) {
    return (
      <div className="result-section">
        <div className="result-empty">
          <span className="result-empty-icon">📋</span>
          <span className="result-empty-text">Submit a workflow to see results</span>
        </div>
      </div>
    );
  }

  const typedResult = result as { ok: boolean; data?: Record<string, unknown>; error?: string; timing_ms?: number };

  if (!typedResult.ok) {
    return (
      <div className="result-section">
        <div className="result-error">
          <div className="result-error-title">Error</div>
          <div className="result-error-text">{typedResult.error || 'Unknown error'}</div>
        </div>
      </div>
    );
  }

  const yamlContent = typedResult.data?.yaml_text || typedResult.data?.yaml || typedResult.data?.optimized_yaml || '';
  const timing = typedResult.timing_ms || 0;

  return (
    <div className="result-section">
      <div className="result-success">
        <div className="result-header">
          <span className="result-badge">✓ Success</span>
          <span className="result-timing">{timing}ms</span>
        </div>
        <div className="result-content">
          <pre>
            {typeof yamlContent === 'string' && yamlContent 
              ? yamlContent 
              : JSON.stringify(typedResult.data, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}