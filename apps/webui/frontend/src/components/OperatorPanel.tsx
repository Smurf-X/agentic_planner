import { useState, useEffect } from 'react';
import { listOps, explainOp } from '../api/client';

interface OperatorPanelProps {}

export function OperatorPanel({}: OperatorPanelProps) {
  const [operators, setOperators] = useState<string[]>([]);
  const [selectedOp, setSelectedOp] = useState<string | null>(null);
  const [opDetails, setOpDetails] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listOps()
      .then((resp) => {
        if (resp.ok && resp.data.operators) {
          setOperators(resp.data.operators as string[]);
        } else {
          setError(resp.error || 'Failed to list operators');
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'List failed'))
      .finally(() => setLoading(false));
  }, []);

  const handleExplain = async (opName: string) => {
    setSelectedOp(opName);
    setLoading(true);
    setError(null);
    try {
      const resp = await explainOp(opName);
      if (resp.ok) {
        setOpDetails(resp.data);
      } else {
        setError(resp.error || 'Failed to explain operator');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Explain failed');
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: '16px', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h3 style={{ marginBottom: '12px' }}>Operators</h3>
      {loading && !operators.length && <p>Loading operators...</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <div style={{ maxHeight: '200px', overflow: 'auto' }}>
        {operators.map((op) => (
          <div
            key={op}
            onClick={() => handleExplain(op)}
            style={{
              padding: '8px',
              cursor: 'pointer',
              background: selectedOp === op ? '#e0e0e0' : 'transparent',
              borderBottom: '1px solid #eee',
            }}
          >
            {op}
          </div>
        ))}
      </div>
      {selectedOp && opDetails && (
        <div style={{ marginTop: '16px', padding: '12px', background: '#f5f5f5', borderRadius: '4px' }}>
          <h4>{selectedOp}</h4>
          <pre style={{ overflow: 'auto', maxHeight: '150px' }}>
            {JSON.stringify(opDetails, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}