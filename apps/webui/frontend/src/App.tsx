import { useState } from 'react';
import { WorkflowForm } from './components/WorkflowForm';
import { ResultPanel } from './components/ResultPanel';
import { OperatorPanel } from './components/OperatorPanel';
import { ValidatePanel } from './components/ValidatePanel';

type Tab = 'workflow' | 'operators' | 'validate';

function App() {
  const [tab, setTab] = useState<Tab>('workflow');
  const [result, setResult] = useState<unknown | null>(null);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <h1 style={{ marginBottom: '24px' }}>Agentic Planner WebUI</h1>
      
      <div style={{ marginBottom: '16px' }}>
        <button onClick={() => setTab('workflow')} style={{ marginRight: '8px', fontWeight: tab === 'workflow' ? 'bold' : 'normal' }}>
          Workflow
        </button>
        <button onClick={() => setTab('operators')} style={{ marginRight: '8px', fontWeight: tab === 'operators' ? 'bold' : 'normal' }}>
          Operators
        </button>
        <button onClick={() => setTab('validate')} style={{ fontWeight: tab === 'validate' ? 'bold' : 'normal' }}>
          Validate
        </button>
      </div>

      {tab === 'workflow' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <WorkflowForm onResult={setResult} />
          <ResultPanel result={result} />
        </div>
      )}

      {tab === 'operators' && <OperatorPanel />}

      {tab === 'validate' && <ValidatePanel />}
    </div>
  );
}

export default App;