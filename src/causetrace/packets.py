"""Build the model input for a target event: SAFE fields plus the gated factual narrative.

This is the only builder of model input for a target event. It selects an explicit
allowlist of SAFE columns (docs/DATA_LEAKAGE.md) and ``narr_accp_safe``, the factual
narrative after the leakage gate. It must never import ``causetrace.gold`` or read
GOLD_ONLY data; tests check this module's source.
"""

from datetime import date

from pydantic import BaseModel, ConfigDict

from causetrace.db import Connection, connect


class _Packet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Location(_Packet):
    city: str | None
    state: str | None
    country: str | None
    latitude: float | None
    longitude: float | None
    nearest_airport_id: str | None
    airport_name: str | None


class Weather(_Packet):
    basic_conditions: str | None  # VMC / IMC
    light_condition: str | None
    ceiling_condition: str | None
    ceiling_height_ft: int | None
    visibility_sm: float | None
    temperature: int | None
    dew_point: int | None
    wind_direction_deg: int | None
    wind_speed_kts: int | None
    gust_kts: int | None
    altimeter_inhg: float | None


class Injuries(_Packet):
    highest: str | None
    fatal: int | None
    serious: int | None
    minor: int | None
    none: int | None


class AircraftFacts(_Packet):
    aircraft_key: int
    category: str | None
    make: str | None
    model: str | None
    series: str | None
    phase_of_flight: str | None
    damage: str | None
    narr_accp: str | None  # factual narrative after the leakage gate; None = withheld


class IncidentPacket(_Packet):
    ev_id: str
    ev_date: date
    ev_time: int | None  # local HHMM
    ev_timezone: str | None
    location: Location
    weather: Weather
    injuries: Injuries
    aircraft: list[AircraftFacts]


_EVENT_SQL = """
SELECT ev_id, ev_date, ev_time, ev_tmzn,
       ev_city, ev_state, ev_country, dec_latitude, dec_longitude, ev_nr_apt_id, apt_name,
       wx_cond_basic, light_cond, sky_cond_ceil, sky_ceil_ht, vis_sm, wx_temp, wx_dew_pt,
       wind_dir_deg, wind_vel_kts, gust_kts, altimeter,
       ev_highest_injury, inj_tot_f, inj_tot_s, inj_tot_m, inj_tot_n
FROM events
WHERE ev_id = %s
"""

_AIRCRAFT_SQL = """
SELECT a.aircraft_key, a.acft_category, a.acft_make, a.acft_model, a.acft_series,
       a.phase_flt_spec, a.damage, n.narr_accp_safe
FROM aircraft a
LEFT JOIN narratives n ON n.ev_id = a.ev_id AND n.aircraft_key = a.aircraft_key
WHERE a.ev_id = %s
ORDER BY a.aircraft_key
"""


def get_incident_packet(ev_id: str, conn: Connection | None = None) -> IncidentPacket:
    """Return the packet for ``ev_id``; raise ``LookupError`` if the event is unknown."""
    if conn is None:
        with connect() as own:
            return get_incident_packet(ev_id, own)
    ev = conn.execute(_EVENT_SQL, (ev_id,)).fetchone()
    if ev is None:
        raise LookupError(f"unknown event {ev_id!r}")
    aircraft = conn.execute(_AIRCRAFT_SQL, (ev_id,)).fetchall()
    return IncidentPacket(
        ev_id=ev[0],
        ev_date=ev[1],
        ev_time=ev[2],
        ev_timezone=ev[3],
        location=Location(
            city=ev[4],
            state=ev[5],
            country=ev[6],
            latitude=ev[7],
            longitude=ev[8],
            nearest_airport_id=ev[9],
            airport_name=ev[10],
        ),
        weather=Weather(
            basic_conditions=ev[11],
            light_condition=ev[12],
            ceiling_condition=ev[13],
            ceiling_height_ft=ev[14],
            visibility_sm=ev[15],
            temperature=ev[16],
            dew_point=ev[17],
            wind_direction_deg=ev[18],
            wind_speed_kts=ev[19],
            gust_kts=ev[20],
            altimeter_inhg=ev[21],
        ),
        injuries=Injuries(highest=ev[22], fatal=ev[23], serious=ev[24], minor=ev[25], none=ev[26]),
        aircraft=[
            AircraftFacts(
                aircraft_key=a[0],
                category=a[1],
                make=a[2],
                model=a[3],
                series=a[4],
                phase_of_flight=a[5],
                damage=a[6],
                narr_accp=a[7],
            )
            for a in aircraft
        ],
    )
