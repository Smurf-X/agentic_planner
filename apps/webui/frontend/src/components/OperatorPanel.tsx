import { useState, useEffect } from 'react';
import { listOps, explainOp } from '../api/client';

interface OperatorPanelProps {}

export function OperatorPanel({}: OperatorPanelProps) {
  const [operators, setOperators] = useState<string[]>([]);
  const [filteredOps, setFilteredOps] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedOp, setSelectedOp] = useState<string | null>(null);
  const [opDetails, setOpDetails] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    listOps()
      .then((resp) => {
        if (resp.ok && resp.data.operators) {
          const ops = resp.data.operators as string[];
          setOperators(ops);
          setFilteredOps(ops);
        } else {
          setError(resp.error || 'Failed to list operators');
        }
      })
      .catch((e) => setError(e instanceof Error ? e.message : 'List failed'))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (searchTerm) {
      setFilteredOps(operators.filter(op => 
        op.toLowerCase().includes(searchTerm.toLowerCase())
      ));
    } else {
      setFilteredOps(operators);
    }
  }, [searchTerm, operators]);

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
    <div className="operator-section">
      <div className="operator-header">
        <h3 className="form-title">Available Operators</h3>
        <span className="operator-count">{operators.length} operators</span>
      </div>

      <div className="operator-search">
        <input
          type="text"
          className="form-input"
          placeholder="Search operators..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      {error && (
        <div className="result-error">
          <div className="result-error-text">{error}</div>
        </div>
      )}

      {loading && !operators.length ? (
        <div className="result-empty">
          <span className="loading">
            <span className="spinner"></span>
            Loading operators...
          </span>
        </div>
      ) : (
        <div className="operator-list">
          {filteredOps.map((op) => (
            <div
              key={op}
              className={`operator-item ${selectedOp === op ? 'selected' : ''}`}
              onClick={() => handleExplain(op)}
            >
              {op}
            </div>
          ))}
        </div>
      )}

      {selectedOp && opDetails && (
        <div className="operator-detail">
          <div className="operator-detail-header">
            <span className="operator-detail-name">{selectedOp}</span>
            {opDetails.category && (
              <span className="operator-detail-category">
                {String(opDetails.category)}
              </span>
            )}
          </div>
          
          <div className="operator-detail-body">
            {opDetails.summary && (
              <p>{String(opDetails.summary)}</p>
            )}
            
            {opDetails.tags && Array.isArray(opDetails.tags) && opDetails.tags.length > 0 && (
              <div className="operator-detail-section">
                <div className="operator-detail-label">Tags</div>
                <div className="operator-tags">
                  {opDetails.tags.map((tag: string) => (
                    <span key={tag} className="operator-tag">{tag}</span>
                  ))}
                </div>
              </div>
            )}

            {opDetails.signature && (
              <div className="operator-detail-section">
                <div className="operator-detail-label">Signature</div>
                <div className="result-content">
                  <pre>{String(opDetails.signature)}</pre>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}