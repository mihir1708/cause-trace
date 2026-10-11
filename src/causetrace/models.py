"""Pydantic models for rows of the NTSB ``mdb-export`` CSVs.

Field names are the lowercased CSV headers, which are also the PostgreSQL column names.
Blank cells become ``None``; dates look like ``10/01/26 07:36:00`` (MM/DD/YY HH:MM:SS).
"""

import re
from datetime import date, datetime
from typing import Annotated, Any, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

# Two-digit years below the pivot are 20xx, the rest 19xx. NTSB records start in 1962.
CENTURY_PIVOT = 50

_MDB_DATETIME = re.compile(r"^(\d{2})/(\d{2})/(\d{2}|\d{4})(?: (\d{2}):(\d{2}):(\d{2}))?$")


def parse_mdb_datetime(value: str) -> datetime:
    """Parse an ``mdb-export`` timestamp such as ``10/01/26 07:36:00``."""
    m = _MDB_DATETIME.match(value.strip())
    if not m:
        raise ValueError(f"expected MM/DD/YY HH:MM:SS, got {value!r}")
    month, day, year_s = int(m[1]), int(m[2]), m[3]
    year = int(year_s)
    if len(year_s) == 2:
        year += 2000 if year < CENTURY_PIVOT else 1900
    hour, minute, second = (int(x) if x else 0 for x in (m[4], m[5], m[6]))
    return datetime(year, month, day, hour, minute, second)


def _blank_to_none(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


def _to_datetime(value: Any) -> Any:
    value = _blank_to_none(value)
    return parse_mdb_datetime(value) if isinstance(value, str) else value


def _to_date(value: Any) -> Any:
    value = _to_datetime(value)
    return value.date() if isinstance(value, datetime) else value


def _to_bool(value: Any) -> Any:
    value = _blank_to_none(value)
    if isinstance(value, str):
        if value.upper() in {"T", "TRUE", "Y"}:
            return True
        if value.upper() in {"F", "FALSE", "N"}:
            return False
        raise ValueError(f"expected T or F, got {value!r}")
    return value


OptStr = Annotated[str | None, BeforeValidator(_blank_to_none)]
OptInt = Annotated[int | None, BeforeValidator(_blank_to_none)]
OptFloat = Annotated[float | None, BeforeValidator(_blank_to_none)]
OptDateTime = Annotated[datetime | None, BeforeValidator(_to_datetime)]
OptBool = Annotated[bool | None, BeforeValidator(_to_bool)]
ReqDate = Annotated[date, BeforeValidator(_to_date)]
EvId = Annotated[str, BeforeValidator(_blank_to_none), Field(pattern=r"^[0-9]{8}[0-9A-Z][0-9]{5}$")]
TwoDigits = Annotated[str, BeforeValidator(_blank_to_none), Field(pattern=r"^[0-9]{2}$")]
AircraftKey = Annotated[int, BeforeValidator(_blank_to_none), Field(ge=1)]


class Row(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class EventRow(Row):
    ev_id: EvId
    ntsb_no: OptStr = None
    ev_type: OptStr = None
    ev_date: ReqDate
    ev_year: OptInt = None
    ev_time: OptInt = None
    ev_tmzn: OptStr = None
    ev_city: OptStr = None
    ev_state: OptStr = None
    ev_country: OptStr = None
    dec_latitude: OptFloat = None
    dec_longitude: OptFloat = None
    ev_nr_apt_id: OptStr = None
    apt_name: OptStr = None
    wx_cond_basic: OptStr = None
    light_cond: OptStr = None
    sky_cond_ceil: OptStr = None
    sky_ceil_ht: OptInt = None
    vis_sm: OptFloat = None
    wx_temp: OptInt = None
    wx_dew_pt: OptInt = None
    wind_dir_deg: OptInt = None
    wind_vel_kts: OptInt = None
    gust_kts: OptInt = None
    altimeter: OptFloat = None
    ev_highest_injury: OptStr = None
    inj_tot_f: OptInt = None
    inj_tot_s: OptInt = None
    inj_tot_m: OptInt = None
    inj_tot_n: OptInt = None
    lchg_date: OptDateTime = None

    @model_validator(mode="after")
    def _year_matches_date(self) -> Self:
        if self.ev_year is not None and self.ev_year != self.ev_date.year:
            raise ValueError(f"ev_year {self.ev_year} does not match ev_date {self.ev_date}")
        return self


class AircraftRow(Row):
    ev_id: EvId
    aircraft_key: AircraftKey = Field(validation_alias="Aircraft_Key")
    regis_no: OptStr = None
    ntsb_no: OptStr = None
    far_part: OptStr = None
    damage: OptStr = None
    acft_category: OptStr = None
    acft_make: OptStr = None
    acft_model: OptStr = None
    acft_series: OptStr = None
    phase_flt_spec: OptStr = None
    homebuilt: OptStr = None
    num_eng: OptInt = None
    lchg_date: OptDateTime = None


class NarrativeRow(Row):
    ev_id: EvId
    aircraft_key: AircraftKey = Field(validation_alias="Aircraft_Key")
    narr_accp: OptStr = None
    narr_accf: OptStr = None
    narr_cause: OptStr = None
    narr_inc: OptStr = None
    lchg_date: OptDateTime = None


class FindingRow(Row):
    ev_id: EvId
    aircraft_key: AircraftKey = Field(validation_alias="Aircraft_Key")
    finding_no: int
    finding_code: OptStr = None
    finding_description: OptStr = None
    category_no: TwoDigits
    subcategory_no: TwoDigits
    section_no: TwoDigits
    subsection_no: OptStr = None
    modifier_no: OptStr = None
    cause_factor: OptStr = Field(default=None, validation_alias="Cause_Factor")
    cm_inpc: OptBool = Field(default=None, validation_alias="cm_inPc")
    lchg_date: OptDateTime = None

    @property
    def label_code(self) -> str:
        return f"{self.category_no}-{self.subcategory_no}-{self.section_no}"

    @property
    def label_short_name(self) -> str | None:
        """First three ``-``-separated segments of the description (category-sub-section)."""
        if not self.finding_description:
            return None
        return "-".join(self.finding_description.split("-")[:3]).strip()
