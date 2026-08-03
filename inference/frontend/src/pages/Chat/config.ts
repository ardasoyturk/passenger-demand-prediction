/**
 * Browser-safe chat configuration.
 *
 * Do not add API keys here: this file is bundled and sent to every browser.
 * The Gateway credential is configured only on the FastAPI server.
 */
export type LLMProvider = 'gateway' | 'openai-compatible';

export const CHAT_CONFIG: {
	provider: LLMProvider;
	model: string;
	gatewayBaseUrl: string;
	openAICompatibleBaseUrl: string;
	mcpUrl: string;
} = {
	provider: 'gateway',
	model: 'stepfun/step-3.5-flash',
	gatewayBaseUrl: '/api/gateway',
	openAICompatibleBaseUrl: 'http://localhost:1234/v1',
	mcpUrl: '/api/mcp/',
};
