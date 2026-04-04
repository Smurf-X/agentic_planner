interface ResultPanelProps {
  result: unknown | null;
}

export function ResultPanel({ result }: ResultPanelProps) {
  if (!result) {
    return (
      <div style={{ padding: '16px', border: '1px solid #ccc', borderRadius: '8px', background: '#f9f9f9' }}>
        <p style={{ color: '#666' }}>No result yet. Submit a workflow to see output.</p>
      </div>
    );
  }

  const typedResult = result as { ok: boolean; data?: Record<string, unknown>; error?: string; timing_ms?: number };

  if (!typedResult.ok) {
    return (
      <div style={{ padding: '16px', border: '1px solid red', borderRadius: '8px', background: '#fff0f0' }}>
        <h3 style={{ color: 'red' }}>Error</h3>
        <p>{typedResult.error || 'Unknown error'}</p>
      </div>
    );
  }

  const yamlContent = typedResult.data?.yaml || typedResult.data?.optimized_yaml || '';
  const timing = typedResult.timing_ms || 0;

  return (
    <div style={{ padding: '16px', border: '1px solid #4caf50', borderRadius: '8px', background: '#f0fff0' }}>
      <h3 style={{ color: '#4caf50' }}>Result</h3>
      <p style={{ fontSize: '12px', color: '#666' }}>Completed in {timing}ms</p>
      {yamlContent && (
        <pre style={{ background: '#fff', padding: '12px', borderRadius: '4px', overflow: 'auto', maxHeight: '300px' }}>
          {typeof yamlContent === 'string' ? yamlContent : JSON.stringify(yamlContent, null, 2)}
        </pre>
      )}
      {!yamlContent && typedResult.data && (
        <pre style={{ background: '#fff', padding: '12px', borderRadius: '4px', overflow: 'auto', maxHeight: '300px' }}>
          {JSON.stringify(typedResult.data, null, 2)}
        </pre>
      )}
    </div>
  );
}