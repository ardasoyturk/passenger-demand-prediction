import { createMCPClient } from '@ai-sdk/mcp';
import { CHAT_CONFIG } from './config';

// The AI SDK MCP HTTP transport constructs a URL object itself, so it cannot
// accept a relative path. Resolving it here keeps the default same-origin
// through Vite's /api proxy while also accepting an absolute deployment URL.
const MCP_URL = new URL(
	CHAT_CONFIG.mcpUrl,
	window.location.origin,
).toString();

/** Open one stateless HTTP MCP client for a single model response. */
export function createPredictionMCPClient() {
	return createMCPClient({
		transport: {
			type: 'http',
			url: MCP_URL,
			// The browser's native fetch needs Window as its receiver. Supplying a
			// bound function prevents the SDK's stored function reference from
			// throwing "Illegal invocation" when it performs MCP requests.
			fetch: window.fetch.bind(window),
		},
	});
}
