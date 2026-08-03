import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { createGateway } from 'ai';

// ─────────────────────────────────────────────────────────────────────────────
//  LLM ENDPOINT YAPILANDIRMASI
//
//  Varsayılan sağlayıcı Vercel AI Gateway'dir. OpenAI-compatible herhangi bir
//  sunucu (vLLM, LM Studio, Ollama, LiteLLM …) için VITE_LLM_PROVIDER değerini
//  "openai-compatible" yapabilirsiniz.
//
//  Değerler `.env` üzerinden de verilebilir (Vite):
//      VITE_LLM_PROVIDER=gateway
//      VITE_LLM_MODEL=anthropic/claude-sonnet-4.5
//
//  OpenAI-compatible örneği:
//      VITE_LLM_PROVIDER=openai-compatible
//      VITE_LLM_BASE_URL=http://sunucu:8000/v1
//      VITE_LLM_MODEL=qwen3-32b
//      VITE_LLM_API_KEY=…             (endpoint kimlik doğrulaması isterse)
// ─────────────────────────────────────────────────────────────────────────────

export type LLMProvider = 'gateway' | 'openai-compatible';

const configuredProvider = import.meta.env.VITE_LLM_PROVIDER as string | undefined;
export const LLM_PROVIDER: LLMProvider = configuredProvider === 'openai-compatible'
	? 'openai-compatible'
	: 'gateway';

// Gateway model kimliği `sağlayıcı/model` biçiminde olmalıdır.
// TODO: Kullanacağınız Gateway modelini buraya girin veya VITE_LLM_MODEL kullanın.
export const LLM_MODEL: string =
	(import.meta.env.VITE_LLM_MODEL as string | undefined) ?? 'alibaba/qwen3.7-flash';

// TODO: OpenAI-compatible endpoint adresini buraya girin.
const LLM_BASE_URL: string =
	(import.meta.env.VITE_LLM_BASE_URL as string | undefined) ?? 'http://localhost:1234/v1';

// Gateway key is injected only by FastAPI. The browser sends AI SDK requests
// to this same-origin API endpoint and never sees the secret.
const GATEWAY_BASE_URL = '/api/gateway';

// OpenAI-compatible endpoint API anahtarı istemiyorsa Authorization eklenmez.
const LLM_API_KEY: string | undefined = import.meta.env.VITE_LLM_API_KEY as string | undefined;

const gateway = createGateway({
	baseURL: GATEWAY_BASE_URL,
});

const openaiCompatible = createOpenAICompatible({
	name: 'inference-llm',
	baseURL: LLM_BASE_URL,
	...(LLM_API_KEY ? { apiKey: LLM_API_KEY } : {}),
});

export const chatModel = LLM_PROVIDER === 'gateway'
	? gateway(LLM_MODEL)
	: openaiCompatible.chatModel(LLM_MODEL);
