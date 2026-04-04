import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';

describe('App Settings tab', () => {
  it('renders Settings tab in navigation', () => {
    render(<App />);
    
    expect(screen.getByRole('button', { name: /Settings/ })).toBeInTheDocument();
  });

  it('Settings tab has settings icon', () => {
    render(<App />);
    
    const settingsTab = screen.getByRole('button', { name: /Settings/ });
    expect(settingsTab.textContent).toContain('⚙️');
  });

  it('shows Settings page when Settings tab is clicked', () => {
    render(<App />);
    
    const settingsTab = screen.getByRole('button', { name: /Settings/ });
    fireEvent.click(settingsTab);
    
    const headings = screen.getAllByRole('heading', { name: 'Settings' });
    expect(headings.length).toBeGreaterThan(0);
  });

  it('Settings tab is included in the tabs array', () => {
    render(<App />);
    
    const workflowTab = screen.getByRole('button', { name: /Workflow/ });
    const operatorsTab = screen.getByRole('button', { name: /Operators/ });
    const validateTab = screen.getByRole('button', { name: /Validate/ });
    const settingsTab = screen.getByRole('button', { name: /Settings/ });
    
    expect(workflowTab).toBeInTheDocument();
    expect(operatorsTab).toBeInTheDocument();
    expect(validateTab).toBeInTheDocument();
    expect(settingsTab).toBeInTheDocument();
  });
});