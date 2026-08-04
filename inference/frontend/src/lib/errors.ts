import { ApiError } from '../api';

export const STOP_ADDITION_ERROR_LABELS: Record<string, string> = {
	COMPANY_NOT_FOUND: 'Firma bulunamadı.',
	CURRENT_ROUTE_NOT_FOUND_FOR_COMPANY: 'Bu firmaya ait mevcut güzergâh bulunamadı.',
	CANDIDATE_STOP_NOT_FOUND: 'Eklenecek durak bulunamadı.',
	CURRENT_ROUTE_TOO_SHORT: 'Mevcut rota durak eklemek için çok kısa.',
	CANDIDATE_STOP_ALREADY_IN_ROUTE: 'Seçilen durak bu rotada zaten bulunuyor.',
	ROUTE_OR_CANDIDATE_COORDINATES_UNUSABLE: 'Rota veya durak koordinatları kullanılamıyor.',
	NO_USABLE_INTERMEDIATE_INSERTION: 'Durak için kullanılabilir bir ara ekleme konumu bulunamadı.',
};

export function errorMessage(error: unknown, labels?: Record<string, string>): string {
	const candidate = error as {
		message?: unknown;
		status?: unknown;
		statusCode?: unknown;
		responseBody?: unknown;
	};
	const status = typeof candidate.statusCode === 'number'
		? candidate.statusCode
		: typeof candidate.status === 'number' ? candidate.status : undefined;

	if (status === 429) return 'AI hizmeti şu anda yoğun. Lütfen birkaç saniye sonra tekrar deneyin.';
	if (status === 502) return 'AI hizmetine ulaşılamadı. Sunucu bağlantısını kontrol edip tekrar deneyin.';
	if (status === 503) return 'AI hizmeti şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin.';
	if (status !== undefined && status >= 500) return 'AI sunucusunda geçici bir sorun oluştu. Lütfen tekrar deneyin.';

	if (error instanceof ApiError) {
		return labels?.[error.message] ?? `İstek tamamlanamadı (${error.status}).`;
	}
	if (typeof candidate.message === 'string' && candidate.message.trim()) {
		return `İstek tamamlanamadı: ${candidate.message}`;
	}
	return 'İstek tamamlanamadı. Lütfen tekrar deneyin.';
}
