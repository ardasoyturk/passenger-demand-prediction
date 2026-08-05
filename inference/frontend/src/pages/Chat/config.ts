/**
 * Browser-safe chat configuration.
 *
 * Do not add API keys here: this file is bundled and sent to every browser.
 * Credentials for both server-side model relays are configured only on the
 * FastAPI server.
 */
export type LLMProvider = 'gateway' | 'openai-compatible';

export const CHAT_CONFIG: {
	provider: LLMProvider;
	model: string;
	gatewayBaseUrl: string;
	openAICompatibleBaseUrl: string;
	mcpUrl: string;
} = {
	provider: 'openai-compatible',
	model: 'inclusionai/ling-3.0-flash:free',
	gatewayBaseUrl: '/api/gateway',
	openAICompatibleBaseUrl: 'https://openrouter.ai/api/v1',
	mcpUrl: '/api/mcp/',
};
