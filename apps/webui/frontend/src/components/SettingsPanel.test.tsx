import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SettingsPanel } from './SettingsPanel';
import { ModelConfig } from '../api/client';

vi.mock('../api/client', () => ({
  testLLM: vi.fn(),
}));

import { testLLM } from '../api/client';

describe('SettingsPanel', () => {
  const mockModelConfig: ModelConfig = {
    base_url: '',
    api_key: '',
    model: '',
  };

  const defaultProps = {
    modelConfig: mockModelConfig,
    llmTested: false,
    onModelConfigChange: vi.fn(),
    onLLMTestedChange: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the settings panel with LLM Configuration section', () => {
    render(<SettingsPanel {...defaultProps} />);
    
    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getByText('LLM Configuration')).toBeInTheDocument();
  });

  it('shows base URL input field', () => {
    render(<SettingsPanel {...defaultProps} />);
    
    expect(screen.getByPlaceholderText('https://api.openai.com/v1 (leave empty for OpenAI)')).toBeInTheDocument();
  });

  it('shows API Key input field with password type', () => {
    render(<SettingsPanel {...defaultProps} />);
    
    const apiKeyInput = screen.getByPlaceholderText('sk-...');
    expect(apiKeyInput).toBeInTheDocument();
    expect(apiKeyInput).toHaveAttribute('type', 'password');
  });

  it('shows Model Name input field', () => {
    render(<SettingsPanel {...defaultProps} />);
    
    expect(screen.getByPlaceholderText('gpt-4o-mini')).toBeInTheDocument();
  });

  it('shows Test Connection button', () => {
    render(<SettingsPanel {...defaultProps} />);
    
    expect(screen.getByRole('button', { name: 'Test Connection' })).toBeInTheDocument();
  });

  it('calls onModelConfigChange when Base URL changes', () => {
    const onModelConfigChange = vi.fn();
    render(<SettingsPanel {...defaultProps} onModelConfigChange={onModelConfigChange} />);
    
    const baseUrlInput = screen.getByPlaceholderText('https://api.openai.com/v1 (leave empty for OpenAI)');
    fireEvent.change(baseUrlInput, { target: { value: 'https://api.example.com/v1' } });
    
    expect(onModelConfigChange).toHaveBeenCalledWith({
      ...mockModelConfig,
      base_url: 'https://api.example.com/v1',
    });
  });

  it('calls onModelConfigChange when API Key changes', () => {
    const onModelConfigChange = vi.fn();
    const onLLMTestedChange = vi.fn();
    render(<SettingsPanel {...defaultProps} onModelConfigChange={onModelConfigChange} onLLMTestedChange={onLLMTestedChange} />);
    
    const apiKeyInput = screen.getByPlaceholderText('sk-...');
    fireEvent.change(apiKeyInput, { target: { value: 'sk-test-key' } });
    
    expect(onModelConfigChange).toHaveBeenCalledWith({
      ...mockModelConfig,
      api_key: 'sk-test-key',
    });
    expect(onLLMTestedChange).toHaveBeenCalledWith(false);
  });

  it('calls onModelConfigChange when Model Name changes', () => {
    const onModelConfigChange = vi.fn();
    const onLLMTestedChange = vi.fn();
    render(<SettingsPanel {...defaultProps} onModelConfigChange={onModelConfigChange} onLLMTestedChange={onLLMTestedChange} />);
    
    const modelInput = screen.getByPlaceholderText('gpt-4o-mini');
    fireEvent.change(modelInput, { target: { value: 'gpt-4o-mini' } });
    
    expect(onModelConfigChange).toHaveBeenCalledWith({
      ...mockModelConfig,
      model: 'gpt-4o-mini',
    });
    expect(onLLMTestedChange).toHaveBeenCalledWith(false);
  });

  it('calls testLLM when Test Connection button is clicked with valid config', async () => {
    const mockTestLLM = vi.mocked(testLLM);
    mockTestLLM.mockResolvedValueOnce({
      ok: true,
      data: { model: 'gpt-4o-mini' },
      timing_ms: 100,
    });

    const configWithValues: ModelConfig = {
      base_url: 'https://api.openai.com/v1',
      api_key: 'sk-test-key',
      model: 'gpt-4o-mini',
    };

    const onLLMTestedChange = vi.fn();
    render(<SettingsPanel {...defaultProps} modelConfig={configWithValues} onLLMTestedChange={onLLMTestedChange} />);
    
    const testButton = screen.getByRole('button', { name: 'Test Connection' });
    fireEvent.click(testButton);
    
    expect(mockTestLLM).toHaveBeenCalledWith(configWithValues);
    
    await waitFor(() => {
      expect(onLLMTestedChange).toHaveBeenCalledWith(true);
    });
    
    expect(screen.getByText(/Connection successful/)).toBeInTheDocument();
  });

  it('shows error message when test fails', async () => {
    const mockTestLLM = vi.mocked(testLLM);
    mockTestLLM.mockResolvedValueOnce({
      ok: false,
      data: {},
      timing_ms: 100,
      error: 'Invalid API key',
    });

    const configWithValues: ModelConfig = {
      base_url: '',
      api_key: 'sk-invalid',
      model: 'gpt-4',
    };

    const onLLMTestedChange = vi.fn();
    render(<SettingsPanel {...defaultProps} modelConfig={configWithValues} onLLMTestedChange={onLLMTestedChange} />);
    
    const testButton = screen.getByRole('button', { name: 'Test Connection' });
    fireEvent.click(testButton);
    
    await waitFor(() => {
      expect(screen.getByText(/Invalid API key/)).toBeInTheDocument();
    });
    
    expect(onLLMTestedChange).toHaveBeenCalledWith(false);
  });

  it('shows validation error when testing without API key and model', () => {
    render(<SettingsPanel {...defaultProps} />);
    
    const testButton = screen.getByRole('button', { name: 'Test Connection' });
    fireEvent.click(testButton);
    
    expect(screen.getByText(/Please fill in Model Name and API Key/)).toBeInTheDocument();
    expect(testLLM).not.toHaveBeenCalled();
  });

  it('shows Test Again button when LLM is tested', () => {
    const configWithValues: ModelConfig = {
      base_url: '',
      api_key: 'sk-test',
      model: 'gpt-4',
    };

    render(<SettingsPanel {...defaultProps} modelConfig={configWithValues} llmTested={true} />);
    
    expect(screen.getByRole('button', { name: 'Test Again' })).toBeInTheDocument();
  });

  it('shows status indicator when configured', () => {
    const configWithValues: ModelConfig = {
      base_url: '',
      api_key: 'sk-test',
      model: 'gpt-4o-mini',
    };

    render(<SettingsPanel {...defaultProps} modelConfig={configWithValues} llmTested={true} />);
    
    expect(screen.getByText('LLM: Configured')).toBeInTheDocument();
    expect(screen.getByText('gpt-4o-mini')).toBeInTheDocument();
  });
});