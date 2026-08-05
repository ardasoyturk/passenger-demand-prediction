import { stepCountIs, streamText } from 'ai';
import type { ModelMessage } from 'ai';
import { Bot, Check, Copy, LoaderCircle, Send, Sparkles, Square, Trash2, TriangleAlert } from 'lucide-preact';
import type { TargetedEvent } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { Markdown } from '../../components/Markdown';
import { RouteMap } from '../../components/RouteMap';
import type { RouteDurak } from '../../api';
import { chatModel } from './llm';
import { createPredictionMCPClient } from './mcp';
import { getSystemPrompt } from './system-prompt';
import { errorMessage } from '../../lib/errors';
import { CHAT_CONFIG } from './config';

interface ChatMessage {
	id: string;
	role: 'user' | 'assistant';
	content: string;
	maps?: ChatMap[];
}

interface ChatMap {
	id: string;
	title: string;
	description: string;
	duraklar: RouteDurak[];
	highlightedStopId?: number;
}

const SUGGESTIONS: string[] = [
	'Talep etiketleri (çok düşük, düşük, orta, yüksek) ne anlama geliyor?',
	'42185 firmasının 110926 güzergâhındaki Cuma 13:00 seferini değerlendir.',
	'42185 firmasının 62182 güzergâhına 151 numaralı durağı Cuma 13:00 için eklersek etkisini tahmin et.',
	'Durak eklemenin talep ve mesafe üzerindeki etkisini nasıl yorumlamalıyım?',
	'Eşik olasılıkları ile beklenen talep arasındaki farkı açıkla.',
];

const COMPACT_SUGGESTIONS: string[] = [
	'Talep etiketleri ne anlama geliyor?',
	'Bir seferin talebini değerlendirelim',
	'Durak eklemenin etkisini analiz et',
];

// The compact assistant and /chat are two views of the same conversation.
// sessionStorage keeps that handoff within the current browser tab only.
const CHAT_HISTORY_STORAGE_KEY = 'yolcu-talep-chat-history';

let idCounter = 0;
function messageId(): string {
	idCounter += 1;
	return `msg-${Date.now()}-${idCounter}`;
}

export function Chat({ compact = false }: { compact?: boolean }) {
	const [messages, setMessages] = useState<ChatMessage[]>(readStoredMessages);
	const [input, setInput] = useState('');
	const [streaming, setStreaming] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [responseStartId, setResponseStartId] = useState<string | null>(null);
	const abortRef = useRef<AbortController | null>(null);
	const scrollRef = useRef<HTMLDivElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const pinnedRef = useRef(true);

	// Kullanıcı en alta yakınsa yeni içerikte kaydırmaya devam et.
	useEffect(() => {
		const el = scrollRef.current;
		if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
	}, [messages, streaming, error]);

	// A long answer (especially one with a map) must open at its first line.
	// Further output is only followed when the user has deliberately remained
	// at the bottom of the conversation.
	useEffect(() => {
		if (!responseStartId) return;
		const scrollArea = scrollRef.current;
		const message = scrollArea?.querySelector<HTMLElement>(`[data-message-id="${responseStartId}"]`);
		if (scrollArea && message) {
			const offset = message.getBoundingClientRect().top - scrollArea.getBoundingClientRect().top;
			scrollArea.scrollTo({ top: Math.max(0, scrollArea.scrollTop + offset - 20) });
			pinnedRef.current = false;
		}
		setResponseStartId(null);
	}, [messages, responseStartId]);

	useEffect(() => () => abortRef.current?.abort(), []);

	useEffect(() => {
		try {
			window.sessionStorage.setItem(CHAT_HISTORY_STORAGE_KEY, JSON.stringify(messages));
		} catch {
			// Storage can be unavailable in private or heavily restricted contexts.
		}
	}, [messages]);

	useEffect(() => {
		if (compact) textareaRef.current?.focus();
	}, [compact]);

	function handleScroll() {
		const el = scrollRef.current;
		if (!el) return;
		pinnedRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
	}

	async function send(text: string, base?: ChatMessage[]) {
		const content = text.trim();
		if (!content || streaming) return;
		const history: ChatMessage[] = [...(base ?? messages), { id: messageId(), role: 'user', content }];
		const assistantId = messageId();
		setMessages([...history, { id: assistantId, role: 'assistant', content: '' }]);
		setInput('');
		resetTextarea();
		setError(null);
		setStreaming(true);
		pinnedRef.current = false;
		setResponseStartId(assistantId);
		const controller = new AbortController();
		abortRef.current = controller;
		let mcpClient: Awaited<ReturnType<typeof createPredictionMCPClient>> | undefined;
		try {
			mcpClient = await createPredictionMCPClient();
			const tools = await mcpClient.tools();
			const result = streamText({
				model: chatModel,
				instructions: getSystemPrompt(),
				tools,
				stopWhen: stepCountIs(5),
				messages: history
					.filter((m) => m.content.trim() !== '')
					.map<ModelMessage>((m) => (m.role === 'user'
						? { role: 'user', content: m.content }
						: { role: 'assistant', content: m.content })),
				abortSignal: controller.signal,
				maxRetries: 1,
				temperature: 0.2
			});
			let streamError: unknown;
			for await (const part of result.fullStream) {
				if (part.type === 'text-delta') {
					setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + part.text } : m)));
				} else if (part.type === 'error') {
					streamError = part.error;
				}
			}
			if (streamError !== undefined) {
				throw streamError;
			}
			const maps = mapsFromToolResults(await result.toolResults);
			if (maps.length > 0) {
				setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, maps } : m)));
			}
		} catch (err) {
			if (!controller.signal.aborted) {
				setError(errorMessage(err));
				// Boş kalan asistan yer tutucusunu temizle.
				setMessages((prev) => prev.filter((m) => m.id !== assistantId || m.content.trim() !== ''));
			}
		} finally {
			await mcpClient?.close();
			setStreaming(false);
			abortRef.current = null;
			textareaRef.current?.focus();
		}
	}

	function stop() {
		abortRef.current?.abort();
	}

	function clear() {
		abortRef.current?.abort();
		setMessages([]);
		setError(null);
	}

	function retry() {
		if (streaming) return;
		let idx = -1;
		for (let i = messages.length - 1; i >= 0; i -= 1) {
			if (messages[i]?.role === 'user') {
				idx = i;
				break;
			}
		}
		setError(null);
		if (idx === -1) return;
		const base = messages.slice(0, idx);
		setMessages(base);
		void send(messages[idx]!.content, base);
	}

	function autogrow() {
		const el = textareaRef.current;
		if (!el) return;
		el.style.height = 'auto';
		el.style.height = `${Math.min(el.scrollHeight, 192)}px`;
	}

	function resetTextarea() {
		const el = textareaRef.current;
		if (el) el.style.height = 'auto';
	}

	function handleKeyDown(event: TargetedEvent<HTMLTextAreaElement, KeyboardEvent>) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			void send(input);
		}
	}

	return (
		<div class={`flex min-h-0 flex-1 flex-col ${compact ? 'chat-compact' : ''}`}>
			{!compact && <div class="border-b border-border bg-white">
				<div class="mx-auto flex w-full max-w-3xl items-center gap-3 px-4 py-3 sm:px-6">
					<span class="grid size-8 shrink-0 place-items-center rounded-md bg-primary text-white">
						<Sparkles class="size-4" aria-hidden="true" />
					</span>
					<div class="min-w-0">
						<h1 class="truncate text-sm font-semibold">Yapay Zekâ Asistanı</h1>
						<h2 class="truncate text-xs font-light text-gray-700">{CHAT_CONFIG.model}</h2>
					</div>
					{messages.length > 0 && (
						<button
							type="button"
							onClick={clear}
							class="ml-auto inline-flex h-8 shrink-0 items-center gap-2 rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
						>
							<Trash2 class="size-3.5" aria-hidden="true" />
							<span class="hidden sm:inline">Sohbeti temizle</span>
						</button>
					)}
				</div>
			</div>}

			<div ref={scrollRef} onScroll={handleScroll} class="min-h-0 flex-1 overflow-y-auto">
				{messages.length === 0 ? (
					<EmptyState compact={compact} onSuggestion={(text) => void send(text)} />
				) : (
					<div class="mx-auto w-full max-w-3xl space-y-6 px-4 py-6 sm:px-6">
						{messages.map((message, index) => (
							<MessageBubble
								key={message.id}
								message={message}
								active={streaming && index === messages.length - 1 && message.role === 'assistant'}
							/>
						))}
					</div>
				)}
			</div>

			<div class="shrink-0">
				<div class={`mx-auto w-full max-w-3xl ${compact ? 'px-3 pb-3' : 'px-4 pb-4 sm:px-6 sm:pb-5'}`}>
					{error && <ErrorBanner message={error} onRetry={retry} />}
					<form
						onSubmit={(event) => {
							event.preventDefault();
							void send(input);
						}}
						class="flex items-end gap-2 rounded-lg border border-border bg-white p-2 shadow-sm transition-shadow focus-within:ring-2 focus-within:ring-slate-400/25"
					>
						<textarea
							ref={textareaRef}
							value={input}
							onInput={(event) => {
								setInput(event.currentTarget.value);
								autogrow();
							}}
							onKeyDown={handleKeyDown}
							rows={1}
							placeholder="AI asistana bir soru yazın..."
							aria-label="Mesajınız"
							class="max-h-48 min-h-10 flex-1 resize-none bg-transparent px-2 py-2 text-sm leading-relaxed outline-none placeholder:text-muted-foreground"
						/>
						{streaming ? (
							<button
								type="button"
								onClick={stop}
								aria-label="Yanıtı durdur"
								title="Yanıtı durdur"
								class="grid size-10 shrink-0 place-items-center rounded-md bg-primary text-white shadow-sm transition-colors hover:bg-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900"
							>
								<Square class="size-4" aria-hidden="true" />
							</button>
						) : (
							<button
								type="submit"
								disabled={input.trim() === ''}
								aria-label="Gönder"
								title="Gönder"
								class="grid size-10 shrink-0 place-items-center rounded-md bg-primary text-white shadow-sm transition-colors hover:bg-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-slate-900 disabled:cursor-not-allowed disabled:opacity-40"
							>
								<Send class="size-4" aria-hidden="true" />
							</button>
						)}
					</form>
					{!compact && <p class="mt-2 text-center text-xs text-muted-foreground">
						Enter gönderir, Shift+Enter yeni satır ekler.
					</p>}
				</div>
			</div>
		</div>
	);
}

function readStoredMessages(): ChatMessage[] {
	try {
		const stored = window.sessionStorage.getItem(CHAT_HISTORY_STORAGE_KEY);
		if (!stored) return [];
		const parsed: unknown = JSON.parse(stored);
		if (!Array.isArray(parsed)) return [];
		return parsed.filter((message): message is ChatMessage => (
			isRecord(message)
			&& typeof message.id === 'string'
			&& (message.role === 'user' || message.role === 'assistant')
			&& typeof message.content === 'string'
		));
	} catch {
		return [];
	}
}

function EmptyState({ compact, onSuggestion }: { compact: boolean; onSuggestion: (text: string) => void }) {
	if (compact) {
		return (
			<div class="flex min-h-full flex-col px-5 py-6">
				<div class="grid size-10 place-items-center rounded-lg bg-primary text-white shadow-sm">
					<Sparkles class="size-4.5" aria-hidden="true" />
				</div>
				<h2 class="mt-4 text-base font-semibold tracking-tight">Nasıl yardımcı olabilirim?</h2>
				<p class="mt-1 max-w-sm text-sm leading-6 text-muted-foreground">
					Sefer talebi ve güzergâh kararları için verilerinizi birlikte inceleyelim.
				</p>
				<div class="mt-6">
					<p class="mb-2 text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground">Hızlı başlangıç</p>
					<div class="grid gap-2">
						{COMPACT_SUGGESTIONS.map((suggestion) => (
							<button
								key={suggestion}
								type="button"
								onClick={() => onSuggestion(suggestion)}
								class="group flex items-center gap-3 rounded-lg border border-border bg-white px-3.5 py-3 text-left text-sm font-medium text-foreground shadow-sm transition-all hover:-translate-y-px hover:border-slate-300 hover:shadow-md"
							>
								<span class="size-1.5 shrink-0 rounded-full bg-slate-300 transition-colors group-hover:bg-primary" />
								<span>{suggestion}</span>
							</button>
						))}
					</div>
				</div>
			</div>
		);
	}

	return (
		<div class="mx-auto flex h-full w-full max-w-3xl flex-col items-center justify-center px-4 py-10 text-center sm:px-6">
			<div class="grid size-12 place-items-center rounded-full bg-muted text-muted-foreground">
				<Sparkles class="size-5" aria-hidden="true" />
			</div>
			<h2 class="mt-4 text-lg font-semibold tracking-tight">Size nasıl yardımcı olabilirim?</h2>
			<p class="mt-1 max-w-md text-sm text-muted-foreground">
				Sefer talebi, güzergâh analizi ve model çıktıları hakkındaki sorularınızı yanıtlayabilirim.
			</p>
			<div class="mt-6 flex flex-wrap items-center justify-center gap-2">
				{SUGGESTIONS.map((suggestion) => (
					<button
						key={suggestion}
						type="button"
						onClick={() => onSuggestion(suggestion)}
						class="rounded-full border border-border bg-white px-3.5 py-1.5 text-left text-xs font-medium text-muted-foreground shadow-sm transition-colors hover:bg-muted hover:text-foreground sm:text-sm"
					>
						{suggestion}
					</button>
				))}
			</div>
		</div>
	);
}

function MessageBubble({ message, active }: { message: ChatMessage; active: boolean }) {
	const displayedContent = useTypewriterText(message.content, active && message.role === 'assistant');
	if (message.role === 'user') {
		return (
			<div data-message-id={message.id} class="flex justify-end">
				<div class="max-w-[85%] whitespace-pre-wrap break-words rounded-lg bg-muted px-4 py-2.5 text-sm leading-relaxed sm:max-w-[75%]">
					{message.content}
				</div>
			</div>
		);
	}

	const waiting = active && displayedContent === '';
	const isTyping = displayedContent.length < message.content.length;
	return (
		<div data-message-id={message.id} class="flex gap-3">
			<span class="mt-0.5 grid size-7 shrink-0 place-items-center rounded-md bg-primary text-white">
				<Bot class="size-4" aria-hidden="true" />
			</span>
			<div class="min-w-0 flex-1 pt-0.5">
				{waiting ? (
					<span class="inline-flex items-center gap-2 text-sm text-muted-foreground" role="status">
						<LoaderCircle class="size-4 animate-spin" aria-hidden="true" />
						Yanıt oluşturuluyor
					</span>
				) : (
					<>
						<div class={isTyping || active ? 'chat-response is-streaming' : 'chat-response'}>
							<Markdown content={displayedContent} />
							{(isTyping || active) && <span class="chat-caret" aria-hidden="true" />}
						</div>
						{!active && message.maps?.map((map) => (
							<div key={map.id} class="mt-4">
								<RouteMap
									duraklar={map.duraklar}
									highlightedStopId={map.highlightedStopId}
									title={map.title}
									description={map.description}
								/>
							</div>
						))}
						{!active && !isTyping && message.content !== '' && <MessageActions text={message.content} />}
					</>
				)}
			</div>
		</div>
	);
}

function mapsFromToolResults(toolResults: Awaited<ReturnType<typeof streamText>>['toolResults'] extends PromiseLike<infer T> ? T : never): ChatMap[] {
	const maps: ChatMap[] = [];
	const proposedMaps: ChatMap[] = [];
	for (const result of toolResults) {
		if (result.type !== 'tool-result') continue;
		if (
			result.toolName === 'predict_stop_addition_demand'
			|| result.toolName === 'predict_general_stop_addition_demand'
		) {
			const proposal = stopAdditionRoute(result.output);
			if (proposal && proposal.duraklar.length > 0) {
				proposedMaps.push({
					id: result.toolCallId,
					title: 'Önerilen güzergâh haritası',
					description: 'Yeni durak yeşil renkle vurgulanır',
					duraklar: proposal.duraklar,
					highlightedStopId: proposal.addedStopId,
				});
			}
		}
		if (
			result.toolName === 'get_company_route_details'
			|| result.toolName === 'get_canonical_route_details'
			|| result.toolName === 'get_route_details'
		) {
			const duraklar = routeStops(result.output);
			if (duraklar.length > 0) {
				maps.push({
					id: result.toolCallId,
					title: 'Güzergâh haritası',
					description: 'Araçtan alınan sıralı güzergâh konumları',
					duraklar,
				});
			}
		}
		if (result.toolName === 'get_stop_details') {
			const stop = stopFromResult(result.output);
			if (stop) {
				maps.push({
					id: result.toolCallId,
					title: 'Durak konumu',
					description: stop.durak_adi ?? `Durak ${stop.durak_id}`,
					duraklar: [stop],
					highlightedStopId: stop.durak_id,
				});
			}
		}
	}
	// A stop-addition answer may also call route details for context. In that
	// case the proposed route is the useful map and must replace the old route.
	return proposedMaps.length > 0 ? proposedMaps : maps;
}

function stopAdditionRoute(value: unknown): { duraklar: RouteDurak[]; addedStopId?: number } | null {
	value = unwrapMcpToolOutput(value);
	if (!isRecord(value) || !Array.isArray(value.proposed_route_stops)) return null;
	const duraklar = value.proposed_route_stops.flatMap((stop) => {
		const parsed = routeStopFromResult(stop);
		return parsed ? [parsed] : [];
	});
	return {
		duraklar,
		addedStopId: isFiniteNumber(value.added_stop_uetds_yer_id)
			? value.added_stop_uetds_yer_id
			: undefined,
	};
}

function routeStops(value: unknown): RouteDurak[] {
	value = unwrapMcpToolOutput(value);
	if (!isRecord(value)) return [];
	// The unscoped route lookup may resolve a company route (`route`) or a
	// canonical route (`canonical_route`); both retain the standard stop shape.
	const route = isRecord(value.route)
		? value.route
		: isRecord(value.canonical_route)
			? value.canonical_route
			: value;
	if (!Array.isArray(route.duraklar)) return [];
	return route.duraklar.flatMap((stop) => {
		const parsed = routeStopFromResult(stop);
		return parsed ? [parsed] : [];
	});
}

function stopFromResult(value: unknown): RouteDurak | null {
	value = unwrapMcpToolOutput(value);
	if (!isRecord(value)) return null;
	return routeStopFromResult({
		sira: 1,
		durak_id: value.id,
		durak_adi: value.uetds_adi,
		kisa_adi: value.kisa_adi,
		il_id: value.il_id,
		ilce_id: value.ilce_id,
		enlem: value.enlem,
		boylam: value.boylam,
	});
}

function routeStopFromResult(value: unknown): RouteDurak | null {
	if (!isRecord(value) || !isFiniteNumber(value.sira) || !isFiniteNumber(value.durak_id)) return null;
	return {
		sira: value.sira,
		durak_id: value.durak_id,
		durak_adi: nullableString(value.durak_adi),
		kisa_adi: nullableString(value.kisa_adi),
		il_id: nullableNumber(value.il_id),
		ilce_id: nullableNumber(value.ilce_id),
		enlem: nullableNumber(value.enlem),
		boylam: nullableNumber(value.boylam),
	};
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === 'object' && value !== null;
}

/** MCP's dynamically discovered tools return the protocol result wrapper. */
function unwrapMcpToolOutput(value: unknown): unknown {
	if (!isRecord(value)) return value;
	if (isRecord(value.structuredContent)) return value.structuredContent;
	if (!Array.isArray(value.content)) return value;
	const textPart = value.content.find((part): part is { type: 'text'; text: string } => (
		isRecord(part) && part.type === 'text' && typeof part.text === 'string'
	));
	if (!textPart) return value;
	try {
		return JSON.parse(textPart.text) as unknown;
	} catch {
		return value;
	}
}

function isFiniteNumber(value: unknown): value is number {
	return typeof value === 'number' && Number.isFinite(value);
}

function nullableNumber(value: unknown): number | null {
	return isFiniteNumber(value) ? value : null;
}

function nullableString(value: unknown): string | null {
	return typeof value === 'string' ? value : null;
}

/** Smooths network-sized stream chunks into a readable typewriter cadence. */
function useTypewriterText(content: string, isActive: boolean): string {
	const [visible, setVisible] = useState(content);
	const visibleRef = useRef(content);
	const wasActiveRef = useRef(isActive);

	useEffect(() => {
		// Existing, completed messages should never animate when the chat rerenders.
		if (!isActive && !wasActiveRef.current) {
			visibleRef.current = content;
			setVisible(content);
			return;
		}

		let timer: number | undefined;
		const reveal = () => {
			const current = visibleRef.current.length;
			if (current >= content.length) return;
			// Larger incoming chunks are consumed gradually without making the UI lag.
			const characters = Math.max(1, Math.ceil((content.length - current) / 14));
			const next = content.slice(0, current + characters);
			visibleRef.current = next;
			setVisible(next);
			timer = window.setTimeout(reveal, 18);
		};
		reveal();
		wasActiveRef.current = isActive;
		return () => { if (timer !== undefined) window.clearTimeout(timer); };
	}, [content, isActive]);

	return visible;
}

function MessageActions({ text }: { text: string }) {
	const [copied, setCopied] = useState(false);

	async function copy() {
		try {
			await navigator.clipboard.writeText(text);
			setCopied(true);
			window.setTimeout(() => setCopied(false), 1600);
		} catch {
			// Panoya erişilemedi; sessizce geç.
		}
	}

	return (
		<div class="mt-2 flex items-center gap-1">
			<button
				type="button"
				onClick={copy}
				class="inline-flex h-7 items-center gap-1.5 rounded-md px-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
			>
				{copied ? <Check class="size-3.5 text-emerald-600" aria-hidden="true" /> : <Copy class="size-3.5" aria-hidden="true" />}
				{copied ? 'Kopyalandı' : 'Kopyala'}
			</button>
		</div>
	);
}

function ErrorBanner({ message, onRetry }: { message: string; onRetry: () => void }) {
	return (
		<div class="animate-enter mb-3 flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700" role="alert">
			<TriangleAlert class="mt-0.5 size-4 shrink-0" aria-hidden="true" />
			<div>
				<strong class="font-medium">Yanıt alınamadı.</strong> <span>{message}</span>{' '}
				<button type="button" onClick={onRetry} class="font-medium underline underline-offset-2 hover:text-red-800">
					Tekrar dene
				</button>
			</div>
		</div>
	);
}
