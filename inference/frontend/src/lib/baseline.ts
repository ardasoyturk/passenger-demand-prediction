import type { GeneralPrediction, ReliabilityEvidence } from '../api';

export function baselineExplanation(prediction: GeneralPrediction | ReliabilityEvidence) {
	if ('baseline_trip_count' in prediction) {
		const count = prediction.baseline_trip_count.toLocaleString('tr-TR');
		switch (prediction.baseline_source) {
			case 'company_route':
				return `Aynı firma ve güzergâhtaki ${count} geçmiş seferin ortalaması kullanıldı.`;
			case 'canonical_route':
				return `Aynı fiziksel rotadaki ${count} geçmiş seferin ortalaması kullanıldı.`;
			case 'global':
				return 'Daha özel bir geçmiş eşleşmesi bulunamadığı için genel tarihsel ortalama kullanıldı.';
			default:
				return 'Kullanılan tarihsel referans kaynağı belirlenemedi.';
		}
	}

	const format = (count: number) => count.toLocaleString('tr-TR');
	switch (prediction.baseline_source) {
		case 'company_route_time_weekday':
			return `Aynı firma, güzergâh, saat ve hafta günündeki ${format(prediction.exact_time_weekday_count)} geçmiş seferin ortalaması kullanıldı.`;
		case 'company_route_time':
			return `Aynı firma, güzergâh ve saatteki ${format(prediction.exact_time_count)} geçmiş seferin ortalaması kullanıldı.`;
		case 'company_route':
			return `Aynı firma ve güzergâhtaki ${format(prediction.company_route_count)} geçmiş seferin ortalaması kullanıldı.`;
		case 'canonical_route_time_weekday':
			return `Aynı fiziksel rota, zaman dilimi ve hafta günündeki ${format(prediction.canonical_time_weekday_count)} geçmiş seferin ortalaması kullanıldı.`;
		case 'canonical_route':
			return `Aynı fiziksel rotadaki ${format(prediction.canonical_route_count)} geçmiş seferin ortalaması kullanıldı.`;
		case 'global':
			return 'Daha özel bir geçmiş eşleşmesi bulunamadığı için genel tarihsel ortalama kullanıldı.';
		default:
			return 'Kullanılan tarihsel referans kaynağı belirlenemedi.';
	}
}
