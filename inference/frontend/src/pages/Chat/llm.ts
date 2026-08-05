import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createGateway } from 'ai';

import { CHAT_CONFIG } from './config';

export const LLM_PROVIDER = CHAT_CONFIG.provider;
export const LLM_MODEL = CHAT_CONFIG.model;

const gateway = createGateway({
	baseURL: CHAT_CONFIG.gatewayBaseUrl,
});

const openaiCompatible = createOpenAICompatible({
	name: 'inference-llm',
	// Keep the AI SDK in the browser, but send its OpenAI-compatible request to
	// FastAPI. The upstream URL is a public setting; the API key is server-only.
	baseURL: new URL('/api/openai-compatible', window.location.origin).toString(),
	headers: {
		'x-openai-compatible-base-url': CHAT_CONFIG.openAICompatibleBaseUrl,
	},
});

export const chatModel = LLM_PROVIDER === 'gateway'
	? gateway(LLM_MODEL)
	: openaiCompatible.chatModel(LLM_MODEL);
