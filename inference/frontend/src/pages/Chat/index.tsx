import { stepCountIs, streamText } from 'ai';
import type { ModelMessage } from 'ai';
import { Bot, Check, Copy, LoaderCircle, Send, Sparkles, Square, Trash2, TriangleAlert } from 'lucide-preact';
import type { TargetedEvent } from 'preact';
import { useEffect, useRef, useState } from 'preact/hooks';
import { Markdown } from '../../components/Markdown';
import { chatModel } from './llm';
import { createPredictionMCPClient } from './mcp';
import { getSystemPrompt } from './system-prompt';

interface ChatMessage {
	id: string;
	role: 'user' | 'assistant';
	content: string;
}

const SUGGESTIONS: string[] = [
	'Talep etiketleri (çok düşük, düşük, orta, yüksek) ne anlama geliyor?',
	'42185 firmasının 110926 güzergâhındaki Cuma 13:00 seferini değerlendir.',
	'42185 firmasının 62182 güzergâhına 151 numaralı durağı Cuma 13:00 için eklersek etkisini tahmin et.',
	'Durak eklemenin talep ve mesafe üzerindeki etkisini nasıl yorumlamalıyım?',
	'Eşik olasılıkları ile beklenen talep arasındaki farkı açıkla.',
];

let idCounter = 0;
function messageId(): string {
	idCounter += 1;
	return `msg-${Date.now()}-${idCounter}`;
}

export function Chat() {
	const [messages, setMessages] = useState<ChatMessage[]>([]);
	const [input, setInput] = useState('');
	const [streaming, setStreaming] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const abortRef = useRef<AbortController | null>(null);
	const scrollRef = useRef<HTMLDivElement>(null);
	const textareaRef = useRef<HTMLTextAreaElement>(null);
	const pinnedRef = useRef(true);

	// Kullanıcı en alta yakınsa yeni içerikte kaydırmaya devam et.
	useEffect(() => {
		const el = scrollRef.current;
		if (el && pinnedRef.current) el.scrollTop = el.scrollHeight;
	}, [messages, streaming, error]);

	useEffect(() => () => abortRef.current?.abort(), []);

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
		pinnedRef.current = true;
		const controller = new AbortController();
		abortRef.current = controller;
		let mcpClient: Awaited<ReturnType<typeof createPredictionMCPClient>> | undefined;
		try {
			mcpClient = await createPredictionMCPClient();
			const tools = await mcpClient.tools();
			const { textStream } = streamText({
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
			for await (const delta of textStream) {
				setMessages((prev) => prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + delta } : m)));
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
		<div class="flex min-h-0 flex-1 flex-col">
			<div class="border-b border-border bg-white">
				<div class="mx-auto flex w-full max-w-3xl items-center gap-3 px-4 py-3 sm:px-6">
					<span class="grid size-8 shrink-0 place-items-center rounded-md bg-primary text-white">
						<Sparkles class="size-4" aria-hidden="true" />
					</span>
					<div class="min-w-0">
						<h1 class="truncate text-sm font-semibold">Yapay Zekâ Asistanı</h1>
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
			</div>

			<div ref={scrollRef} onScroll={handleScroll} class="min-h-0 flex-1 overflow-y-auto">
				{messages.length === 0 ? (
					<EmptyState onSuggestion={(text) => void send(text)} />
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
				<div class="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6 sm:pb-5">
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
					<p class="mt-2 text-center text-xs text-muted-foreground">
						Enter gönderir, Shift+Enter yeni satır ekler.
					</p>
				</div>
			</div>
		</div>
	);
}

function EmptyState({ onSuggestion }: { onSuggestion: (text: string) => void }) {
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
			<div class="flex justify-end">
				<div class="max-w-[85%] whitespace-pre-wrap break-words rounded-lg bg-muted px-4 py-2.5 text-sm leading-relaxed sm:max-w-[75%]">
					{message.content}
				</div>
			</div>
		);
	}

	const waiting = active && displayedContent === '';
	const isTyping = displayedContent.length < message.content.length;
	return (
		<div class="flex gap-3">
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
						{!active && !isTyping && message.content !== '' && <MessageActions text={message.content} />}
					</>
				)}
			</div>
		</div>
	);
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

function errorMessage(error: unknown): string {
	if (error instanceof Error) return error.message;
	return 'Bilinmeyen bir hata oluştu.';
}
