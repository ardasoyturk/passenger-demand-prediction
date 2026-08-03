import { describe, expect, test } from 'bun:test';
import { parseChatMarkdown } from './chat-markdown';

describe('parseChatMarkdown', () => {
	test('renders inline KaTeX', () => {
		expect(parseChatMarkdown('Olasılık: $P(Y \\geq 20)$')).toContain('class="katex"');
	});

	test('renders block KaTeX', () => {
		const html = parseChatMarkdown('$$\n\\operatorname{MAE} = \\frac{1}{n}\n$$');
		expect(html).toContain('katex-display');
		expect(html).not.toContain('$$');
	});

	test('keeps GFM tables', () => {
		const html = parseChatMarkdown('| Ölçüt | Değer |\n|---|---:|\n| MAE | 8 |');
		expect(html).toContain('<table>');
		expect(html).toContain('<th>Ölçüt</th>');
	});

	test('does not render math inside fenced code', () => {
		const html = parseChatMarkdown('```latex\n$$ x^2 $$\n```');
		expect(html).toContain('<pre><code');
		expect(html).toContain('$$ x^2 $$');
		expect(html).not.toContain('class="katex"');
	});

	test('keeps malformed LaTeX from throwing', () => {
		expect(() => parseChatMarkdown('Broken: $\\frac{1$')).not.toThrow();
	});
});
