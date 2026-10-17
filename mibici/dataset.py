from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime, time, timedelta
from html.parser import HTMLParser
from io import BufferedReader, BytesIO
from pathlib import Path
import re
from typing import Annotated
import unicodedata
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer

from mibici.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

DATA_PAGE_URL = "https://mibici.net/es/datos-abiertos/"
REQUEST_TIMEOUT_SECONDS = 60
CSV_CHUNK_SIZE = 100_000

MONTHLY_FILE_RE = re.compile(r"datos_abiertos_(\d{4})_(\d{2})(?:-[\w-]+)?\.csv")
NOMENCLATURE_FILE_RE = re.compile(r"nomenclatura_(\d{4})_(\d{2})\.csv")
TRIP_COLUMN_NAMES = {
    "viaje_id": "trip_id",
    "usuario_id": "user_id",
    "genero": "gender",
    "ano_de_nacimiento": "birth_year",
    "inicio_del_viaje": "started_at",
    "fin_del_viaje": "ended_at",
    "origen_id": "origin_station_id",
    "destino_id": "destination_station_id",
}
TRIP_COLUMN_ALIASES = {"a_ae_nacimiento": "ano_de_nacimiento"}
TRIP_OUTPUT_COLUMNS = [
    "trip_id",
    "user_id",
    "gender",
    "birth_year",
    "started_at",
    "ended_at",
    "origin_station_id",
    "origin_station_name",
    "origin_latitude",
    "origin_longitude",
    "origin_capacity",
    "destination_station_id",
    "destination_station_name",
    "destination_latitude",
    "destination_longitude",
    "destination_capacity",
]
STATION_FIELDS = {
    "station_name": "station_name",
    "latitude": "latitude",
    "longitude": "longitude",
    "capacity": "capacity",
}

app = typer.Typer()


class DownloadLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.monthly_links: dict[tuple[int, int], str] = {}
        self.nomenclature_links: list[tuple[tuple[int, int], str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        href = dict(attrs).get("href")
        if href is None:
            return

        filename = urlparse(href).path.rsplit("/", maxsplit=1)[-1]
        monthly_match = MONTHLY_FILE_RE.fullmatch(filename)
        nomenclature_match = NOMENCLATURE_FILE_RE.fullmatch(filename)

        if monthly_match:
            year, month = (int(value) for value in monthly_match.groups())
            self.monthly_links[(year, month)] = href
        elif nomenclature_match:
            year, month = (int(value) for value in nomenclature_match.groups())
            self.nomenclature_links.append(((year, month), href))


def _download(url: str) -> bytes:
    request = _request(url)
    with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.read()


def _read_trip_csv_chunks(url: str) -> Iterator[pd.DataFrame]:
    with urlopen(_request(url), timeout=REQUEST_TIMEOUT_SECONDS) as response:
        buffered_response = BufferedReader(response)
        header = buffered_response.peek(8192).splitlines()[0]
        encoding = _detect_csv_encoding(header)
        yield from pd.read_csv(
            buffered_response,
            encoding=encoding,
            chunksize=CSV_CHUNK_SIZE,
        )


def _detect_csv_encoding(header: bytes) -> str:
    try:
        header.decode("utf-8-sig")
    except UnicodeDecodeError:
        return "latin-1"
    return "utf-8-sig"


def _request(url: str) -> Request:
    return Request(url, headers={"User-Agent": "mibici-open-data-downloader/1.0"})


def _same_site_url(href: str, base_url: str) -> str:
    absolute_url = urljoin(base_url, href)
    if urlparse(absolute_url).netloc != urlparse(base_url).netloc:
        raise ValueError(f"Unexpected external download URL: {absolute_url}")
    return absolute_url


def find_download_links(
    html: str, page_url: str = DATA_PAGE_URL
) -> tuple[dict[tuple[int, int], str], str]:
    parser = DownloadLinkParser()
    parser.feed(html)

    monthly_links = {
        month: _same_site_url(href, page_url) for month, href in parser.monthly_links.items()
    }
    if not monthly_links:
        raise ValueError(f"No monthly CSV downloads found on {page_url}")
    if not parser.nomenclature_links:
        raise ValueError(f"No station nomenclature CSV found on {page_url}")

    latest_nomenclature = max(parser.nomenclature_links, key=lambda entry: entry[0])
    nomenclature_url = _same_site_url(latest_nomenclature[1], page_url)
    return monthly_links, nomenclature_url


def _normalize_column_name(column: str) -> str:
    without_accents = unicodedata.normalize("NFKD", column)
    ascii_name = without_accents.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r"[^a-zA-Z0-9]+", "_", ascii_name).strip("_").lower()


def load_station_nomenclature(csv_bytes: bytes) -> pd.DataFrame:
    stations = pd.read_csv(BytesIO(csv_bytes), encoding="utf-8-sig")
    stations.columns = [_normalize_column_name(column) for column in stations.columns]
    stations = stations.rename(
        columns={
            "id": "station_id",
            "name": "station_name",
            "dpcapacity": "capacity",
        }
    )
    required_columns = {
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "capacity",
    }
    missing_columns = required_columns.difference(stations.columns)
    if missing_columns:
        raise ValueError(
            "Station nomenclature is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    stations = stations.loc[:, list(required_columns)].copy()
    stations["station_id"] = pd.to_numeric(stations["station_id"], errors="raise").astype("Int64")
    if stations["station_id"].duplicated().any():
        raise ValueError("Station nomenclature contains duplicate station IDs")
    stations["station_name"] = _normalize_text(stations["station_name"])
    for column in ("latitude", "longitude"):
        stations[column] = pd.to_numeric(stations[column], errors="raise").astype("float64")
    stations["capacity"] = pd.to_numeric(stations["capacity"], errors="raise").astype("Int64")
    return stations


def _normalize_text(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .map(lambda value: unicodedata.normalize("NFC", value) if pd.notna(value) else value)
    )



def parse_trip_dates(values: pd.Series) -> pd.Series:
    """Interpreta fechas con reglas explícitas, sin inferir día y mes.

    Las fechas con barras se interpretan como día/mes/año.
    Las fechas con guiones admitidas comienzan por el año.
    Rechaza valores faltantes, fechas imposibles y formatos desconocidos.
    Conserva la hora local tal como aparece en la fuente, sin convertir a UTC.
    """
    text = values.astype("string").str.strip()
    if text.isna().any() or text.eq("").any():
        raise ValueError("Hay fechas faltantes o vacías.")

    result = pd.Series(
        pd.NaT, index=values.index, dtype="datetime64[ns]", name=values.name
    )
    recognized = pd.Series(False, index=values.index)

    formats = [
        (r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}",
         "%d/%m/%Y %H:%M"),
        (r"\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2}",
         "%d/%m/%Y %H:%M:%S"),
        (r"\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}",
         "%Y-%m-%d %H:%M"),
        (r"\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}",
         "%Y-%m-%d %H:%M:%S"),
        (r"\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}\.\d{1,9}",
         "%Y-%m-%d %H:%M:%S.%f"),
    ]

    for pattern, date_format in formats:
        mask = text.str.fullmatch(pattern)
        if mask.any():
            result.loc[mask] = pd.to_datetime(
                text.loc[mask], format=date_format, errors="raise", exact=True
            ).astype("datetime64[ns]")
            recognized |= mask

    if not recognized.all():
        examples = text.loc[~recognized].head(3).tolist()
        raise ValueError(f"Formato de fecha no contemplado: {examples}")

    return result


def normalize_trip_chunk(chunk: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    trips = chunk.copy()
    trips.columns = [_normalize_column_name(column) for column in trips.columns]
    trips = trips.rename(columns=TRIP_COLUMN_ALIASES)
    trips = trips.rename(columns=TRIP_COLUMN_NAMES)

    required_columns = set(TRIP_COLUMN_NAMES.values())
    missing_columns = required_columns.difference(trips.columns)
    if missing_columns:
        raise ValueError(
            "Trip CSV is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    trips = trips.loc[:, list(TRIP_COLUMN_NAMES.values())].copy()
    for column in ("trip_id", "user_id", "origin_station_id", "destination_station_id"):
        trips[column] = pd.to_numeric(trips[column], errors="raise").astype("Int64")
    trips["birth_year"] = _parse_nullable_integer(trips["birth_year"], "birth_year")
    trips["gender"] = _normalize_text(trips["gender"])
    for column in ("started_at", "ended_at"):
        trips[column] = parse_trip_dates(trips[column])

    station_lookup = stations.set_index("station_id")
    for side, station_id in (
        ("origin", "origin_station_id"),
        ("destination", "destination_station_id"),
    ):
        for output_field, station_field in STATION_FIELDS.items():
            trips[f"{side}_{output_field}"] = trips[station_id].map(station_lookup[station_field])

    return trips.loc[:, TRIP_OUTPUT_COLUMNS]


def _parse_nullable_integer(values: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid_values = values.notna() & numeric.isna()
    if invalid_values.any():
        raise ValueError(f"Column {column_name} contains non-numeric values")
    fractional_values = numeric.notna() & numeric.mod(1).ne(0)
    if fractional_values.any():
        raise ValueError(f"Column {column_name} contains non-integer values")
    return numeric.astype("Int64")


def _subtract_years(value: datetime, years: int) -> datetime:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


def _date_range(start_date: date | None, end_date: date | None) -> tuple[datetime, datetime]:
    now = datetime.now()
    start = datetime.combine(start_date, time.min) if start_date else _subtract_years(now, 3)
    end = datetime.combine(end_date + timedelta(days=1), time.min) if end_date else now
    if start >= end:
        raise ValueError("The start date must be earlier than the end date")
    return start, end


def _months_in_range(start: datetime, end: datetime) -> list[tuple[int, int]]:
    year, month = start.year, start.month
    months: list[tuple[int, int]] = []
    while (year, month) <= (end.year, end.month):
        month_start = datetime(year, month, 1)
        if month_start < end:
            months.append((year, month))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
    return months


def download_and_clean(
    start: datetime,
    end: datetime,
    output_path: Path,
    nomenclature_output_path: Path,
) -> int:
    page_html = _download(DATA_PAGE_URL).decode("utf-8")
    monthly_links, nomenclature_url = find_download_links(page_html)
    nomenclature_bytes = _download(nomenclature_url)
    stations = load_station_nomenclature(nomenclature_bytes)

    current_month = date.today().replace(day=1)
    selected_links: list[tuple[tuple[int, int], str]] = []
    for month in _months_in_range(start, end):
        url = monthly_links.get(month)
        if url is None:
            if date(*month, 1) < current_month:
                raise ValueError(f"MiBici does not publish a CSV for {month[0]}-{month[1]:02d}")
            continue
        selected_links.append((month, url))
    if not selected_links:
        raise ValueError("No published monthly CSVs overlap the requested date range")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nomenclature_output_path.parent.mkdir(parents=True, exist_ok=True)
    nomenclature_output_path.write_bytes(nomenclature_bytes)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    row_count = 0

    try:
        for (year, month), url in selected_links:
            logger.info("Downloading trips for {}-{:02d}", year, month)
            for chunk in _read_trip_csv_chunks(url):
                normalized = normalize_trip_chunk(chunk, stations)

                # Validar antes del filtro de fechas evita ocultar viajes
                # que quedaron desplazados de mes por una conversión incorrecta.
                starts = normalized["started_at"]
                expected_month = starts.dt.year.eq(year) & starts.dt.month.eq(month)
                if not expected_month.all():
                    examples = starts.loc[~expected_month].head(3).astype(str).tolist()
                    raise ValueError(
                        f"CSV {year}-{month:02d}: "
                        f"{int((~expected_month).sum())} inicios fuera del mes. "
                        f"Ejemplos: {examples}"
                    )
                in_range = normalized["started_at"].ge(start) & normalized["started_at"].lt(end)
                normalized = normalized.loc[in_range]
                if normalized.empty:
                    continue

                table = pa.Table.from_pandas(normalized, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_path, table.schema, compression="zstd")
                elif not table.schema.equals(writer.schema, check_metadata=False):
                    raise ValueError(
                        f"Trip CSV schema changed while processing {year}-{month:02d}"
                    )
                writer.write_table(table)
                row_count += len(normalized)

        if writer is None:
            raise ValueError("No trips were found in the requested date range")
        writer.close()
        writer = None
        temporary_path.replace(output_path)
    finally:
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)

    logger.success("Wrote {:,} cleaned trips to {}", row_count, output_path)
    logger.info("Saved station nomenclature to {}", nomenclature_output_path)
    return row_count


@app.command()
def main(
    start_date: Annotated[
        str | None, typer.Option(help="First trip date (YYYY-MM-DD; defaults to three years ago).")
    ] = None,
    end_date: Annotated[
        str | None, typer.Option(help="Last trip date (YYYY-MM-DD; defaults to now).")
    ] = None,
    output_path: Annotated[Path, typer.Option()] = PROCESSED_DATA_DIR / "trips.parquet",
    nomenclature_output_path: Annotated[Path, typer.Option()] = (
        RAW_DATA_DIR / "stations_nomenclature.csv"
    ),
) -> None:
    try:
        parsed_start = date.fromisoformat(start_date) if start_date else None
        parsed_end = date.fromisoformat(end_date) if end_date else None
    except ValueError as error:
        raise typer.BadParameter("Dates must use the YYYY-MM-DD format.") from error

    start, end = _date_range(parsed_start, parsed_end)
    logger.info("Building trip dataset from {} through {}", start, end)
    download_and_clean(start, end, output_path, nomenclature_output_path)


if __name__ == "__main__":
    app()
