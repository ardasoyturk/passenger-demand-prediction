import { useMemo } from 'preact/hooks';
import { renderChatMarkdown } from '../lib/chat-markdown';

import 'highlight.js/styles/github.css';
import 'katex/dist/katex.min.css';

export function Markdown({ content }: { content: string }) {
	const html = useMemo(() => renderChatMarkdown(content), [content]);
	// DOMPurify ile temizlenmiş HTML güvenle basılabilir.
	return <div class="markdown-body" dangerouslySetInnerHTML={{ __html: html }} />;
}
