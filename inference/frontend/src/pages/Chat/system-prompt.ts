import prompt from './system-prompt.txt?raw';

/** Instructions are authored as plain text to keep them easy to review. */
const basePrompt = prompt.trim();

const WEEKDAYS_TR = [
	'Pazar',
	'Pazartesi',
	'Salı',
	'Çarşamba',
	'Perşembe',
	'Cuma',
	'Cumartesi',
];

/** Adds the browser-local calendar date so weekday-only requests are actionable. */
export function getSystemPrompt(now = new Date()): string {
	const date = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
	return `${basePrompt}\n\n## Güncel tarih\nBugünün tarihi ${date}, ${WEEKDAYS_TR[now.getDay()]}. Bu bilgiyi yalnızca tarih çözümlemek için kullan.`;
}
