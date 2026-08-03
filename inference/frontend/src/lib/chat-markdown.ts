import DOMPurify from 'dompurify';
import hljs from 'highlight.js/lib/common';
import { Marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import markedKatex from 'marked-katex-extension';

// One parser instance for the whole application. Extensions must not be added
// from a component render, otherwise Marked registers duplicate tokenizers.
const chatMarkdown = new Marked({ gfm: true, breaks: true });

chatMarkdown.use(markedKatex({ throwOnError: false, nonStandard: true }));
chatMarkdown.use(
	markedHighlight({
		langPrefix: 'hljs language-',
		highlight(code, lang) {
			const language = hljs.getLanguage(lang) ? lang : 'plaintext';
			return hljs.highlight(code, { language }).value;
		},
	}),
);

if (typeof DOMPurify.addHook === 'function') {
	DOMPurify.addHook('afterSanitizeAttributes', (node) => {
		if (node.tagName === 'A') {
			node.setAttribute('target', '_blank');
			node.setAttribute('rel', 'noopener noreferrer');
		}
	});
}

/** Convert GFM and math syntax to HTML. Exported for non-DOM parser tests. */
export function parseChatMarkdown(content: string): string {
	return chatMarkdown.parse(content, { async: false });
}

/** Parse model output as GFM and KaTeX, then sanitize it before DOM injection. */
export function renderChatMarkdown(content: string): string {
	const html = parseChatMarkdown(content);
	return DOMPurify.sanitize(html, {
		USE_PROFILES: { html: true, mathMl: true },
	});
}
