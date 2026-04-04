import { useState } from 'react';
import { WorkflowForm } from './components/WorkflowForm';
import { ResultPanel } from './components/ResultPanel';
import { OperatorPanel } from './components/OperatorPanel';
import { ValidatePanel } from './components/ValidatePanel';
import './styles.css';

type Tab = 'workflow' | 'operators' | 'validate';

const tabs: { id: Tab; label: string; icon: string }[] = [
  { id: 'workflow', label: 'Workflow', icon: '⚡' },
  { id: 'operators', label: 'Operators', icon: '🔧' },
  { id: 'validate', label: 'Validate', icon: '✓' },
];

function App() {
  const [tab, setTab] = useState<Tab>('workflow');
  const [result, setResult] = useState<unknown | null>(null);

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="logo">
            <span className="logo-icon">🤖</span>
            <span className="logo-text">Agentic Planner</span>
          </div>
        </div>
        <nav className="sidebar-nav">
          {tabs.map((t) => (
            <button
              key={t.id}
              className={`nav-item ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              <span className="nav-icon">{t.icon}</span>
              <span className="nav-label">{t.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="status-indicator">
            <span className="status-dot"></span>
            <span>Ready</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="main-header">
          <h1 className="page-title">
            {tabs.find((t) => t.id === tab)?.label}
          </h1>
          <div className="header-actions">
            <button className="btn btn-secondary">
              <span>📖</span> Docs
            </button>
          </div>
        </header>

        <div className="content-area">
          {tab === 'workflow' && (
            <div className="workflow-layout">
              <div className="card workflow-card">
                <WorkflowForm onResult={setResult} />
              </div>
              <div className="card result-card">
                <ResultPanel result={result} />
              </div>
            </div>
          )}

          {tab === 'operators' && (
            <div className="card full-height">
              <OperatorPanel />
            </div>
          )}

          {tab === 'validate' && (
            <div className="card full-height">
              <ValidatePanel />
            </div>
          )}
        </div>
      </main>

      <aside className="auxiliary-bar">
        <div className="aux-section">
          <h3 className="aux-title">Quick Actions</h3>
          <div className="quick-actions">
            <button className="quick-action-btn">
              <span>📄</span> New Pipeline
            </button>
            <button className="quick-action-btn">
              <span>📁</span> Load Config
            </button>
            <button className="quick-action-btn">
              <span>💾</span> Save Session
            </button>
          </div>
        </div>
        <div className="aux-section">
          <h3 className="aux-title">History</h3>
          <div className="history-list">
            <div className="history-item">
              <span className="history-icon">⚡</span>
              <div className="history-content">
                <span className="history-title">Generate Pipeline</span>
                <span className="history-time">2 min ago</span>
              </div>
            </div>
            <div className="history-item">
              <span className="history-icon">🔧</span>
              <div className="history-content">
                <span className="history-title">Optimize Config</span>
                <span className="history-time">5 min ago</span>
              </div>
            </div>
          </div>
        </div>
        <div className="aux-section">
          <h3 className="aux-title">Tips</h3>
          <div className="tip-card">
            <p>Use natural language to describe your data processing pipeline.</p>
          </div>
        </div>
      </aside>
    </div>
  );
}

export default App;