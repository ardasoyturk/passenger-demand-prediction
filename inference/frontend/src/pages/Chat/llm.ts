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
	baseURL: CHAT_CONFIG.openAICompatibleBaseUrl,
});

export const chatModel = LLM_PROVIDER === 'gateway'
	? gateway(LLM_MODEL)
	: openaiCompatible.chatModel(LLM_MODEL);
