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
	// openrouter: model: 'inclusionai/ling-3.0-flash:free',
	model: 'gemini-3.5-flash-lite',
	gatewayBaseUrl: '/api/gateway',
	// openrouter: openAICompatibleBaseUrl: 'https://openrouter.ai/api/v1',
	openAICompatibleBaseUrl: "https://generativelanguage.googleapis.com/v1beta/openai/",
	mcpUrl: '/api/mcp/',
};
