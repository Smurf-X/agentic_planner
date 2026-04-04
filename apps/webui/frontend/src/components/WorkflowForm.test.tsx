import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { WorkflowForm } from './WorkflowForm';
import { ModelConfig } from '../api/client';

vi.mock('../api/client', () => ({
  generate: vi.fn(),
  optimize: vi.fn(),
  testLLM: vi.fn(),
}));

describe('WorkflowForm LLM status indicator', () => {
  const mockModelConfig: ModelConfig = {
    base_url: '',
    api_key: '',
    model: '',
  };

  const defaultProps = {
    onResult: vi.fn(),
    modelConfig: mockModelConfig,
    llmTested: false,
    onModelConfigChange: vi.fn(),
    onLLMTestedChange: vi.fn(),
    onNavigateToSettings: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows compact status indicator instead of full LLM config form', () => {
    render(<WorkflowForm {...defaultProps} />);
    
    expect(screen.queryByPlaceholderText('https://api.openai.com/v1 (leave empty for OpenAI)')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('sk-...')).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText('gpt-4o-mini')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Test Connection' })).not.toBeInTheDocument();
  });

  it('shows "LLM: Not configured" when not tested', () => {
    render(<WorkflowForm {...defaultProps} />);
    
    expect(screen.getByText(/LLM: Not configured/)).toBeInTheDocument();
  });

  it('shows "LLM: Configured" with model name when tested', () => {
    const configWithModel: ModelConfig = {
      base_url: '',
      api_key: 'sk-test',
      model: 'gpt-4o-mini',
    };

    render(<WorkflowForm {...defaultProps} modelConfig={configWithModel} llmTested={true} />);
    
    expect(screen.getByText(/LLM: Configured/)).toBeInTheDocument();
    expect(screen.getByText('gpt-4o-mini')).toBeInTheDocument();
  });

  it('shows Configure button when not configured', () => {
    render(<WorkflowForm {...defaultProps} />);
    
    expect(screen.getByRole('button', { name: /Configure/ })).toBeInTheDocument();
  });

  it('calls onNavigateToSettings when Configure button is clicked', () => {
    const onNavigateToSettings = vi.fn();
    render(<WorkflowForm {...defaultProps} onNavigateToSettings={onNavigateToSettings} />);
    
    const configureButton = screen.getByRole('button', { name: /Configure/ });
    fireEvent.click(configureButton);
    
    expect(onNavigateToSettings).toHaveBeenCalled();
  });

  it('shows status indicator at the top of the form', () => {
    render(<WorkflowForm {...defaultProps} />);
    
    const statusIndicator = screen.getByText(/LLM: Not configured/);
    expect(statusIndicator).toBeInTheDocument();
  });

  it('does not show LLM Configuration section header', () => {
    render(<WorkflowForm {...defaultProps} />);
    
    expect(screen.queryByText('LLM Configuration')).not.toBeInTheDocument();
  });

  it('shows checkmark icon when configured', () => {
    const configWithModel: ModelConfig = {
      base_url: '',
      api_key: 'sk-test',
      model: 'gpt-4o-mini',
    };

    render(<WorkflowForm {...defaultProps} modelConfig={configWithModel} llmTested={true} />);
    
    expect(screen.getByText('✓')).toBeInTheDocument();
  });

  it('status indicator is compact and does not take large space', () => {
    render(<WorkflowForm {...defaultProps} />);
    
    const statusIndicator = screen.getByText(/LLM: Not configured/);
    expect(statusIndicator).toBeInTheDocument();
    expect(statusIndicator.tagName).toBe('SPAN');
  });
});