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
	if (!(error instanceof Error)) return 'Bilinmeyen bir hata oluştu.';
	if (!(error instanceof ApiError)) return error.message;
	return labels?.[error.message] ?? `Sunucu yanıtı: ${error.message}`;
}
