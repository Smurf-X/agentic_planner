import { useState } from 'react';
import { WorkflowForm } from './components/WorkflowForm';
import { ResultPanel } from './components/ResultPanel';

function App() {
  const [result, setResult] = useState<unknown | null>(null);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '20px' }}>
      <h1 style={{ marginBottom: '24px' }}>Agentic Planner WebUI</h1>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
        <WorkflowForm onResult={setResult} />
        <ResultPanel result={result} />
      </div>
    </div>
  );
}

export default App;