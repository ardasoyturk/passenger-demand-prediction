"""Pydantic request and response contracts for the demo API."""

from __future__ import annotations

from datetime import date, time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PredictionRequest(ApiModel):
    firma_id: int = Field(ge=0)
    guzergah_kodu: int = Field(ge=0)
    sefer_tarihi: date
    sefer_saati: time


class ThresholdProbabilities(ApiModel):
    ge_10: float = Field(ge=0.0, le=1.0)
    ge_20: float = Field(ge=0.0, le=1.0)
    ge_30: float = Field(ge=0.0, le=1.0)
    ge_43: float = Field(ge=0.0, le=1.0)


class FrequentDepartureTime(ApiModel):
    departure_time: str = Field(pattern=r"^\d{2}:\d{2}$")
    trip_count: int = Field(ge=1)
    route_share: float = Field(ge=0.0, le=1.0)


class ReliabilityEvidence(ApiModel):
    exact_time_weekday_count: int = Field(ge=0)
    exact_time_count: int = Field(ge=0)
    company_route_count: int = Field(ge=0)
    canonical_time_weekday_count: int = Field(ge=0)
    canonical_route_count: int = Field(ge=0)
    baseline_source: str
    frequent_departure_times: list[FrequentDepartureTime]


class SimplifiedPrediction(ApiModel):
    expected_demand: float
    baseline_demand: float
    demand_label: str
    reliability: str
    reliability_reason: str
    probabilities: ThresholdProbabilities
    reliability_evidence: ReliabilityEvidence


class DetailedPrediction(ApiModel):
    FIRMA_ID: int
    GUZERGAH_KODU: int
    SEFER_TARIHI: date
    SEFER_SAATI: time
    canonical_route_id: int
    v4_1_prediction: float
    weekday_baseline_prediction: float
    v4_2_hybrid_weight: float
    v4_2_hybrid_prediction: float
    probability_ge_10: float
    probability_ge_20: float
    probability_ge_30: float
    probability_ge_43: float
    cutoff_ge_10: float
    cutoff_ge_20: float
    cutoff_ge_30: float
    cutoff_ge_43: float
    classifier_variant_ge_10: str
    classifier_variant_ge_20: str
    classifier_variant_ge_30: str
    classifier_variant_ge_43: str
    v4_4_raw_decision_ge_10: int
    v4_4_raw_decision_ge_20: int
    v4_4_raw_decision_ge_30: int
    v4_4_raw_decision_ge_43: int
    mixed_decision_ge_10: int
    mixed_decision_ge_20: int
    mixed_decision_ge_30: int
    mixed_decision_ge_43: int
    mixed_demand_label: str
    classifier_monotonicity_violation: int
    classifier_monotonicity_correction_applied: int
    prediction_reliability: str
    reliability_reason: str


class Durak(ApiModel):
    id: int
    uetds_kodu: str | None = None
    turu: str | None = None
    uetds_adi: str | None = None
    il_id: int | None = None
    ilce_id: int | None = None
    kisa_adi: str | None = None
    ulke_id: int | None = None
    ulke_adi: str | None = None
    enlem: float | None = None
    boylam: float | None = None


class PaginatedDurakResponse(ApiModel):
    items: list[Durak]
    total: int
    page: int
    page_size: int


class RouteSummary(ApiModel):
    guzergah_kodu: int
    kalkis_durak_adi: str | None = None
    varis_durak_adi: str | None = None


class CompanyRoutesResponse(ApiModel):
    firma_id: int
    routes: list[RouteSummary]


class RouteDurak(ApiModel):
    sira: int
    durak_id: int
    durak_adi: str | None = None
    kisa_adi: str | None = None
    il_id: int | None = None
    ilce_id: int | None = None
    enlem: float | None = None
    boylam: float | None = None


class RouteDetailResponse(ApiModel):
    firma_id: int
    firma_unvan: str | None = None
    guzergah_kodu: int
    canonical_guzergah_id: int
    duraklar: list[RouteDurak]


class HealthResponse(ApiModel):
    status: Literal["ok"]
    database: Literal["ok"]
    artifacts: Literal["loaded"]
