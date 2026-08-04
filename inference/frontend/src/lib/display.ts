type StopLike = {
	durak_adi?: string | null;
	uetds_adi?: string | null;
	kisa_adi?: string | null;
	durak_id?: number;
	id?: number;
};

export function stopName(stop: StopLike | null, fallbackId?: number) {
	return stop?.durak_adi
		?? stop?.uetds_adi
		?? stop?.kisa_adi
		?? `Durak ${stop?.durak_id ?? stop?.id ?? fallbackId}`;
}

export function titleCase(value: string) {
	return value
		.toLocaleLowerCase('tr-TR')
		.split(/\s+/)
		.map((word) => word.charAt(0).toLocaleUpperCase('tr-TR') + word.slice(1))
		.join(' ');
}
