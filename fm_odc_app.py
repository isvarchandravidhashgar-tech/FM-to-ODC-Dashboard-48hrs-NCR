"""
FM-to-ODC-Dashboard-48hrs-NCR
--------------------
Daily 2 files upload karo:
  1) Dashboard wali FM ODC file (CSV / XLSX / XLSB / ZIP)
  2) Mapping file (XLSX / CSV)  -> hub_name / Next Location / Lane Type / Lane STD
App apne aap:
  - date-time ke beech ka comma hatata hai ("01 Apr, 2026 08:08:25")
  - Excel wale sab formula columns lagata hai
  - final file (CSV) download ke liye deta hai
Run:  python -m streamlit run fm_odc_app.py --server.maxUploadSize 2048
"""
import datetime as dt_mod
import io
import os
import re
import shutil
import tempfile
import time
import zipfile

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------- settings
CHUNK_ROWS = 100_000  # Larger chunks reduce per-chunk overhead on large CSV files.
# FM ODC source data is limited to Excel columns A:CN (92 columns).
# Any extra source columns after CN (for example CO:CP) are ignored.
BASE_INPUT_COLS = 92

# Summary table settings (Percentage table row order + conditional formatting limits)
PCT_ROW_ORDER = [
    "Connected ON time",
    "FM Miss",
    "Dock Miss Delay connected",
    "Processing Miss-Delay Manifested",
    "Processing Miss-Not Manifested",
    "Report Error",
]
CF_GOOD_CATEGORY = "Connected ON time"
CF_TARGET_PCT = 98.0   # Target for "Connected ON time" (both DCs)
# Who is responsible for each miss (used for the Summary colours)
CF_FM_CATEGORIES = ["FM Miss"]                      # FM side (not in DC team's control)
CF_DATA_CATEGORIES = ["Report Error"]               # dashboard / data problem
CF_TEAM_CATEGORIES = [                              # DC team's own misses
    "Dock Miss Delay connected",
    "Processing Miss-Delay Manifested",
    "Processing Miss-Not Manifested",
]

# in columns me date-time hota hai (comma wali text ho ya serial number)
KNOWN_DT_COLS = [
    "picked_date",
    "recd_at_pickup_hub_date",
    "manifest_creation_at_pickup_hub",
    "manifest_intransit_from_pickup_hub",
    "manifest_received_at_odc_metadata",
    "manifest_intransit_from_odc_metadata",
    "manifest_received_bagging_dc",
    "shipment_received_at_bagging_dc",
    "manifest_ODC_creation_date",
    "manifest_intransit_from_odc",
]
# (1) se (7) checks: Excel wale columns ke naam
CHECK_COLS = [
    ("(1)recd_at_pickup_hub_date", "recd_at_pickup_hub_date"),
    ("(2)manifest_creation_at_pickup_hub", "manifest_creation_at_pickup_hub"),
    ("(3)manifest_intransit_from_pickup_hub", "manifest_intransit_from_pickup_hub"),
    ("(4)manifest_received_at_odc_metadata", "manifest_received_at_odc_metadata"),
    ("(5)manifest_intransit_from_odc_metadata", "manifest_intransit_from_odc_metadata"),
    ("(6)manifest_received_bagging_dc", "manifest_received_bagging_dc"),
    ("(7)shipment_received_at_bagging_dc", "shipment_received_at_bagging_dc"),
]
# naye (formula) columns, Excel wale order me
NEW_COLS = (
    ["Lane", "Type", "Cut_Off", "Processing ", "Processing_cut_off",
     "Picked Data", "M_creation_date", "M_creation_Time", "Remarks (1st Check)",
     "Last_Vehicle Recieved time", "Bucket"]
    + [c[0] for c in CHECK_COLS]
    + ["Final FM MIss (2 Check)", "Report Error (3 Check)", "Date_24Hrs", "Date_46Hrs",
       "24Hrs Cut_off", "48Hrs Cut_off (4 Check)", "FInal (5 Check)", "Final_Remarks"]
)
REQUIRED = KNOWN_DT_COLS + ["destination_hub", "order_status", "first_manifest_status", "Breach_Flag_48h", "in_transit", "bagging_dc_name"]
DATE_PAT = re.compile(r"^\s*\d{1,2} [A-Za-z]{3},? \d{4}")


# ---------------------------------------------------------------- helpers
def cell_to_str(v):
    if v is None:
        return ""
    if isinstance(v, float):
        if np.isnan(v):
            return ""
        return str(int(v)) if v.is_integer() else repr(v)
    if isinstance(v, (dt_mod.datetime, dt_mod.date)):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    return str(v)


def fmt_days_as_time(x):
    """0.25 -> '06:00:00' (NaN -> '-')"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    secs = int(round(x * 86400))
    sign = "-" if secs < 0 else ""
    secs = abs(secs)
    return "%s%02d:%02d:%02d" % (sign, secs // 3600, (secs % 3600) // 60, secs % 60)


def to_days(v):
    """Mapping file ki Lane STD / time value ko din (fraction) me badalta hai. '-' ya galat -> NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (dt_mod.time,)):
        return (v.hour * 3600 + v.minute * 60 + v.second) / 86400
    if isinstance(v, (dt_mod.datetime, pd.Timestamp)):
        return (v.hour * 3600 + v.minute * 60 + v.second) / 86400
    if isinstance(v, pd.Timedelta):
        return v.total_seconds() / 86400
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3) or 0)) / 86400
    try:
        return float(s)
    except ValueError:
        return np.nan


def to_dt(s):
    """Text/serial/ISO -> datetime. Comma yahi hat jata hai."""
    s = s.fillna("").astype(str).str.strip().str.replace(",", "", regex=False)
    out = pd.to_datetime(s, format="%d %b %Y %H:%M:%S", errors="coerce")
    miss = out.isna() & (s != "")
    if miss.any():
        sub = s[miss]
        num = pd.to_numeric(sub, errors="coerce")
        isnum = num.notna()
        if isnum.any():
            out.loc[sub.index[isnum]] = pd.to_datetime(num[isnum], unit="D", origin="1899-12-30")
        rest = sub[~isnum]
        if len(rest):
            out.loc[rest.index] = pd.to_datetime(rest, errors="coerce", dayfirst=True, format="mixed")
    return out


def dt_to_text(d):
    return d.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")


# ---------------------------------------------------------------- readers
def _open_inner(upl):
    """upload -> (kind, file object, compression)"""
    name = getattr(upl, "name", "file").lower()
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(upl)
        names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".gz", ".xlsx", ".xlsb"))]
        if not names:
            raise ValueError("No CSV, XLSX, or XLSB file was found inside the ZIP archive.")
        inner = names[0]
        if inner.lower().endswith(".csv"):
            return "csv", zf.open(inner), None
        if inner.lower().endswith(".gz"):
            return "csv", zf.open(inner), "gzip"
        return ("xlsx" if inner.lower().endswith(".xlsx") else "xlsb"), io.BytesIO(zf.read(inner)), None
    if name.endswith(".csv"):
        return "csv", upl, None
    if name.endswith(".gz"):
        return "csv", upl, "gzip"
    if name.endswith(".xlsx") or name.endswith(".xlsm"):
        return "xlsx", upl, None
    if name.endswith(".xlsb"):
        return "xlsb", upl, None
    raise ValueError("This file type is not supported: " + name)


def iter_chunks(upl, size=CHUNK_ROWS):
    kind, f, comp = _open_inner(upl)
    if kind == "csv":
        for ch in pd.read_csv(f, dtype=str, keep_default_na=False, chunksize=size,
                              usecols=range(BASE_INPUT_COLS),
                              compression=comp or "infer", encoding="utf-8",
                              encoding_errors="replace", engine="c"):
            ch.columns = [str(c).strip() for c in ch.columns]
            yield ch.reset_index(drop=True)
        return
    if kind == "xlsx":
        from openpyxl import load_workbook
        wb = load_workbook(f, read_only=True, data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = ws.iter_rows(values_only=True)
        header = [str(c).strip() if c is not None else "" for c in next(rows)][:BASE_INPUT_COLS]
        buf = []
        for r in rows:
            buf.append([cell_to_str(v) for v in r[:BASE_INPUT_COLS]])
            if len(buf) >= size:
                yield pd.DataFrame(buf, columns=header[: len(buf[0])])
                buf = []
        if buf:
            yield pd.DataFrame(buf, columns=header[: len(buf[0])])
        return
    if kind == "xlsb":
        from pyxlsb import open_workbook
        with open_workbook(f) as wb:
            with wb.get_sheet(1) as sh:
                rows = sh.rows()
                first = next(rows)
                header = [str(c.v).strip() if c.v is not None else "" for c in first][:BASE_INPUT_COLS]
                buf = []
                for r in rows:
                    vals = [cell_to_str(c.v) for c in r[:BASE_INPUT_COLS]]
                    vals += [""] * (len(header) - len(vals))
                    buf.append(vals[: len(header)])
                    if len(buf) >= size:
                        yield pd.DataFrame(buf, columns=header)
                        buf = []
                if buf:
                    yield pd.DataFrame(buf, columns=header)
        return


def read_mapping(upl):
    name = getattr(upl, "name", "").lower()
    if name.endswith(".csv"):
        df = pd.read_csv(upl, dtype=object, keep_default_na=False)
    else:
        engine = "pyxlsb" if name.endswith(".xlsb") else None
        xl = pd.ExcelFile(upl, engine=engine)
        sheet = None
        # Mapping sheet ko hub_name aur Next Location columns se identify karo
        for sn in xl.sheet_names:
            try:
                head = xl.parse(sn, nrows=3, dtype=object)
            except Exception:
                continue
            low = {str(c).strip().lower() for c in head.columns}
            if "hub_name" in low and "next location" in low:
                sheet = sn
                break
        if sheet is None:
            sheet = "Mapping_Lane" if "Mapping_Lane" in xl.sheet_names else xl.sheet_names[0]
        df = xl.parse(sheet, dtype=object)
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------- mapping tables
def normalize_key(v):
    """Normalize lookup keys for case, whitespace, and non-breaking spaces."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return " ".join(str(v).replace("\u00a0", " ").strip().casefold().split())

def normalize_series(values):
    """Vectorized key normalization for large source columns."""
    return (values.astype("string").fillna("")
            .str.replace("\u00a0", " ", regex=False)
            .str.strip().str.casefold().str.replace(r"\s+", " ", regex=True))


def build_lane_tables(mp, processing_days):
    # Required mapping columns: hub_name -> Next Location -> Lane Type / Lane STD
    cols_low = {str(c).strip().casefold(): c for c in mp.columns}

    def pick(name):
        return cols_low.get(name.casefold())

    c_hub = pick("hub_name")
    c_next = pick("Next Location")
    c_type = pick("Lane Type")
    c_std = pick("Lane STD")

    for need, c in (
        ("hub_name", c_hub),
        ("Next Location", c_next),
        ("Lane Type", c_type),
        ("Lane STD", c_std),
    ):
        if c is None:
            raise ValueError("Required column not found in the mapping file: " + need)

    def val(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return str(v).strip()

    # Match Excel VLOOKUP with exact match and first matching row.
    # Step 1: destination_hub / hub_name -> Next Location
    hub_to_lane = {}
    for h, nxt in zip(mp[c_hub], mp[c_next]):
        k = normalize_key(h)
        if k and k not in hub_to_lane:
            hub_to_lane[k] = val(nxt)

    # Step 2: Next Location -> Lane Type and Lane STD
    lane_info = {}
    for nxt, lane_type, std in zip(mp[c_next], mp[c_type], mp[c_std]):
        k = normalize_key(nxt)
        if k and k not in lane_info:
            days = to_days(std)
            lane_info[k] = {
                "type": val(lane_type),
                "cut_days": days,
                "cut_txt": fmt_days_as_time(days),
                "proc_days": days - processing_days if not np.isnan(days) else np.nan,
                "proc_txt": fmt_days_as_time(days - processing_days) if not np.isnan(days) else "-",
            }
    return hub_to_lane, lane_info


def detect_dt_cols(df):
    cols = [c for c in KNOWN_DT_COLS if c in df.columns]
    for c in df.columns:
        if c in cols or c in NEW_COLS:
            continue
        sample = df[c].astype(str).head(2000)
        sample = sample[sample.str.strip() != ""].head(50)
        if len(sample) >= 3 and sample.str.match(DATE_PAT).mean() > 0.8:
            cols.append(c)
    return cols


def process_chunk(df, hub_to_lane, lane_info, dt_cols, processing_days, lastveh_days, stats):
    df = df.drop(columns=[c for c in NEW_COLS if c in df.columns], errors="ignore")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("The following required columns are missing from the dashboard file: " + ", ".join(missing))

    # 1) date-time columns saaf karo
    dts = {}
    for c in dt_cols:
        d = to_dt(df[c])
        bad = ((df[c].astype(str).str.strip() != "") & d.isna()).sum()
        if bad:
            stats["bad_dates"][c] = stats["bad_dates"].get(c, 0) + int(bad)
        dts[c] = d
        df[c] = dt_to_text(d)

    n = len(df)
    # 2) mapping se Lane / Type / Cut_Off
    hub_key = normalize_series(df["destination_hub"])
    lane = hub_key.map(hub_to_lane)
    unmatched = lane.isna()
    if unmatched.any():
        for h, k in df.loc[unmatched, "destination_hub"].value_counts().items():
            stats["unmatched"][h] = stats["unmatched"].get(h, 0) + int(k)
    lane = lane.fillna("#N/A")
    lane_k = normalize_series(lane)
    type_map = {k: v["type"] for k, v in lane_info.items()}
    cut_txt = {k: v["cut_txt"] for k, v in lane_info.items()}
    proc_txt = {k: v["proc_txt"] for k, v in lane_info.items()}
    proc_days = {k: v["proc_days"] for k, v in lane_info.items()}

    out = df
    out["Lane"] = lane
    out["Type"] = lane_k.map(type_map).fillna("#N/A")
    out["Cut_Off"] = lane_k.map(cut_txt).fillna("#N/A")
    out["Processing "] = fmt_days_as_time(processing_days)
    out["Processing_cut_off"] = lane_k.map(proc_txt).fillna("-")
    proc_num = lane_k.map(proc_days).astype(float)

    # 3) Picked Data, M_creation
    picked = dts["picked_date"].dt.normalize()
    p = dts["manifest_ODC_creation_date"]
    out["Picked Data"] = picked.dt.strftime("%Y-%m-%d").fillna("")
    out["M_creation_date"] = p.dt.strftime("%Y-%m-%d").fillna("")
    out["M_creation_Time"] = p.dt.strftime("%H:%M:%S").fillna("")

    # 4) Remarks (1st Check)
    out["Remarks (1st Check)"] = np.where(
        df["Breach_Flag_48h"].astype(str).str.strip().str.lower() == "no breach",
        "Connected On Time", "Breach")

    # 5) Bucket
    out["Last_Vehicle Recieved time"] = fmt_days_as_time(lastveh_days)
    bucket = picked + pd.Timedelta(days=2) + pd.Timedelta(days=lastveh_days)
    out["Bucket"] = dt_to_text(bucket)

    # 6) (1) se (7)
    flags = []
    for new, src in CHECK_COLS:
        d = dts[src]
        f = np.where(d.isna(), "", np.where(d > bucket, "FM Miss", "On Time"))
        out[new] = f
        flags.append(f)
    fl = np.vstack(flags)
    any_miss = (fl == "FM Miss").any(axis=0)
    any_on = (fl == "On Time").any(axis=0)
    out["Final FM MIss (2 Check)"] = np.where(any_miss, "FM Miss", np.where(any_on, "On Time", ""))

    # 7) Report Error (3 Check): N = bagging_dc received, O = shipment received at bagging dc
    n_blank = dts["manifest_received_bagging_dc"].isna()
    o_blank = dts["shipment_received_at_bagging_dc"].isna()
    out["Report Error (3 Check)"] = np.where(
        n_blank & o_blank, "Report Error",
        np.where(~n_blank & ~o_blank, "No Report Error", ""))

    # 8) Date_24Hrs / Date_46Hrs
    proc_td = pd.to_timedelta(proc_num, unit="D")
    d24 = (picked + pd.Timedelta(days=1) + proc_td).dt.round("s")
    d48 = (picked + pd.Timedelta(days=2) + proc_td).dt.round("s")
    out["Date_24Hrs"] = np.where(d24.isna(), "-", d24.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(""))
    out["Date_46Hrs"] = np.where(d48.isna(), "-", d48.dt.strftime("%Y-%m-%d %H:%M:%S").fillna(""))

    # 9) 24Hrs / 48Hrs Cut_off (4 Check)
    def manifest_check(limit, limit_missing):
        res = np.where(p.isna(), "Not Manifested",
                       np.where(limit_missing, "Manifested on time",   # Excel: "-" ke saath compare TRUE aata hai
                                np.where(p <= limit, "Manifested on time", "Delay Manifested")))
        return res

    out["24Hrs Cut_off"] = manifest_check(d24, d24.isna().to_numpy())
    out["48Hrs Cut_off (4 Check)"] = manifest_check(d48, d48.isna().to_numpy())

    # 10) FInal (5 Check)
    is_fm = (df["order_status"].astype(str).str.strip().str.upper() == "IN_MANIFEST") & \
            (df["first_manifest_status"].astype(str).str.strip().str.upper() == "PARTIALLY_CLOSED")
    out["FInal (5 Check)"] = np.where(is_fm, "FM Miss", "FM ON time")

    # 11) Final_Remarks: apply ONLY the user's specified checks in this order.
    # AY column is identified by its stable header name: in_transit.
    # Connected ON time includes shipments up to 48 hours + 30 seconds.
    transit_hours = pd.to_numeric(out["in_transit"].astype(str).str.strip(), errors="coerce")
    connected_on_time = (
        transit_hours.notna()
        & (transit_hours >= 0)
        & (transit_hours <= (48 + (30 / 3600)))
    )
    fm_miss = out["Final FM MIss (2 Check)"].astype(str).str.strip().eq("FM Miss")
    report_error = out["Report Error (3 Check)"].astype(str).str.strip().eq("Report Error")
    cutoff_48 = out["48Hrs Cut_off (4 Check)"].astype(str).str.strip()

    # Priority: Connected ON time -> FM Miss -> Report Error -> 48Hrs Cut_off.
    # No other/intermediate remarks are added to Final_Remarks.
    final_remarks = np.select(
        [connected_on_time,
         fm_miss,
         report_error,
         cutoff_48.eq("Delay Manifested"),
         cutoff_48.eq("Not Manifested"),
         cutoff_48.eq("Manifested on time")],
        ["Connected ON time",
         "FM Miss",
         "Report Error",
         "Processing Miss-Delay Manifested",
         "Processing Miss-Not Manifested",
         "Dock Miss Delay connected"],
        default=""
    )
    out["Final_Remarks"] = final_remarks

    base = [c for c in out.columns if c not in NEW_COLS]
    out = out[base + NEW_COLS]

    # summary
    stats["rows"] += n
    for col in ("FInal (5 Check)", "Final FM MIss (2 Check)", "24Hrs Cut_off", "48Hrs Cut_off (4 Check)"):
        vc = out[col].value_counts()
        d = stats["counts"].setdefault(col, {})
        for k, v in vc.items():
            d[k] = d.get(k, 0) + int(v)
    dc_counts = stats.setdefault("dc_counts", {})
    grouped = out.groupby(["bagging_dc_name", "Final_Remarks"], dropna=False).size()
    for (dc_name, remark), count in grouped.items():
        dc = str(dc_name).strip() if str(dc_name).strip() else "(Blank DC)"
        rm = str(remark).strip() if str(remark).strip() else "(Blank Remark)"
        dc_counts.setdefault(dc, {})[rm] = dc_counts.setdefault(dc, {}).get(rm, 0) + int(count)
    return out


def _total_bytes(f):
    try:
        if hasattr(f, "size"):
            return int(f.size)
        pos = f.tell()
        f.seek(0, 2)
        n = f.tell()
        f.seek(pos)
        return n
    except Exception:
        return 0


def build_dc_summaries(dc_counts):
    """Create Excel-pivot-style DC count and DC-wise percentage tables."""
    categories = [
        "Connected ON time",
        "Dock Miss Delay connected",
        "FM Miss",
        "Processing Miss-Delay Manifested",
        "Processing Miss-Not Manifested",
        "Report Error",
    ]
    # Merge NCR Tauru DC bagging data into NCR Bilaspur DC for both summaries.
    # Keep all other DCs unchanged and remove the separate Tauru column.
    dc_counts = {dc: dict(values) for dc, values in dc_counts.items()}
    bilaspur_key = next(
        (dc for dc in dc_counts if str(dc).strip().casefold() == "ncr bilaspur dc"),
        "NCR Bilaspur DC",
    )
    tauru_key = None
    for dc in dc_counts:
        if str(dc).strip().casefold() == "ncr tauru dc":
            tauru_key = dc
            break
    if tauru_key is not None:
        dc_counts.setdefault(bilaspur_key, {})
        for category, value in dc_counts[tauru_key].items():
            dc_counts[bilaspur_key][category] = (
                dc_counts[bilaspur_key].get(category, 0) + int(value)
            )
        if tauru_key != bilaspur_key:
            del dc_counts[tauru_key]

    dcs = sorted(dc_counts.keys(), key=lambda x: str(x).casefold())
    dc_totals = {dc: sum(dc_counts.get(dc, {}).values()) for dc in dcs}
    grand_total = sum(dc_totals.values())

    count_by_cat = {}
    pct_by_cat = {}
    for category in categories:
        count_row = {"Final_Remarks": category}
        pct_row = {"Final_Remarks": category}
        category_grand = 0
        for dc in dcs:
            value = int(dc_counts.get(dc, {}).get(category, 0))
            category_grand += value
            count_row[dc] = value
            pct_row[dc] = round((value / dc_totals[dc]) * 100, 2) if dc_totals[dc] else 0.0
        count_row["Grand Total"] = category_grand
        pct_row["Grand Total"] = round((category_grand / grand_total) * 100, 2) if grand_total else 0.0
        count_by_cat[category] = count_row
        pct_by_cat[category] = pct_row

    # Count table: biggest Grand Total first (like the Excel pivot).
    # Percentage table: SAME order as the count table, so each row matches its count row.
    count_order = sorted(categories, key=lambda c: (-count_by_cat[c]["Grand Total"], PCT_ROW_ORDER.index(c) if c in PCT_ROW_ORDER else 99))
    pct_order = count_order   # same row order in both tables, so every row lines up
    count_rows = [count_by_cat[c] for c in count_order]
    pct_rows = [pct_by_cat[c] for c in pct_order]

    total_count = {"Final_Remarks": "Grand Total"}
    total_pct = {"Final_Remarks": "Grand Total"}
    for dc in dcs:
        total_count[dc] = dc_totals[dc]
        total_pct[dc] = 100.0 if dc_totals[dc] else 0.0
    total_count["Grand Total"] = grand_total
    total_pct["Grand Total"] = 100.0 if grand_total else 0.0
    count_rows.append(total_count)
    pct_rows.append(total_pct)
    return pd.DataFrame(count_rows), pd.DataFrame(pct_rows)


def _owner(category):
    """Who owns this Final_Remarks row (used for the Owner tag and the bar colour)."""
    if category == CF_GOOD_CATEGORY:
        return "tgt"
    if category in CF_FM_CATEGORIES:
        return "fm"
    if category in CF_DATA_CATEGORIES:
        return "data"
    if category in CF_TEAM_CATEGORIES:
        return "team"
    return ""


def summary_tables_html(count_df, pct_df, target=CF_TARGET_PCT):
    """Management-ready summary: count table + percentage table in ONE white card.
    - Connected ON time row: green if target achieved, soft red if below (with gap in points)
    - Miss rows in the % table: bar = how much of the allowed miss (100 - target) that row uses
    - Owner tag on every row: Target / FM / Dashboard Data / DC Team"""
    import html as _html

    css = (
        "<style>"
        ".fmcard{background:#ffffff;border-radius:10px;padding:18px 22px;box-shadow:0 1px 6px rgba(0,0,0,.30);"
        "overflow-x:auto;}"
        ".fmwrap{display:flex;flex-wrap:wrap;gap:32px;align-items:flex-start;}"
        ".fmtitle{font-family:'Segoe UI',Calibri,Arial,sans-serif;font-size:14px;font-weight:600;color:#1f3864;"
        "margin:0 0 8px 2px;}"
        "table.fms{border-collapse:collapse;font-family:'Segoe UI',Calibri,Arial,sans-serif;font-size:14px;"
        "color:#1f2937;background:#ffffff;}"
        ".fms th{background:#1f3864;color:#ffffff;font-weight:600;padding:9px 10px;text-align:center;"
        "white-space:nowrap;}"
        ".fms th:first-child{text-align:left;}"
        ".fms td{padding:8px 10px;text-align:center;border-bottom:1px solid #e5e7eb;white-space:nowrap;"
        "color:#1f2937;background:#ffffff;position:relative;}"
        ".fms td:first-child{text-align:left;font-weight:500;min-width:215px;}"
        ".fms td.ok{background:#e6f4ea;color:#137333;font-weight:600;}"
        ".fms td.low{background:#fdecea;color:#b3261e;font-weight:600;}"
        ".fms td .bar{position:absolute;left:0;top:0;bottom:0;opacity:.85;}"
        ".fms td .val{position:relative;}"
        ".fms td .gap{position:relative;font-size:11px;font-weight:500;margin-top:1px;}"
        ".fms tr.gt td{background:#eef1f7;font-weight:700;border-top:2px solid #1f3864;border-bottom:none;"
        "color:#1f2937;}"
        ".chip{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600;"
        "white-space:nowrap;}"
        ".chip.tgt{background:#e8eefc;color:#1f3864;}"
        ".chip.fm{background:#fff1e0;color:#b45309;}"
        ".chip.data{background:#fef9c3;color:#854d0e;}"
        ".chip.team{background:#fdecea;color:#b3261e;}"
        "</style>"
    )

    budget = 100.0 - target
    bar_colour = {"fm": "#f6c8a0", "data": "#ffe08a", "team": "#f4a5a0"}
    owner_text = {"tgt": "Target {:.0f}%+".format(target), "fm": "FM", "data": "Dashboard Data", "team": "DC Team"}
    pct_lookup = {r[0]: dict(zip(pct_df.columns[1:], r[1:])) for r in pct_df.itertuples(index=False)}

    def one(df, as_pct, with_owner):
        cols = list(df.columns)
        head = "<th>{}</th>".format(_html.escape(str(cols[0])))
        if with_owner:
            head += "<th>Owner</th>"
        head += "".join("<th>{}</th>".format(_html.escape(str(c))) for c in cols[1:])
        body = []
        last = len(df) - 1
        for idx, row in enumerate(df.itertuples(index=False)):
            category = str(row[0])
            is_total = idx == last
            owner = "" if is_total else _owner(category)
            cells = ["<td>{}</td>".format(_html.escape(category))]
            if with_owner:
                chip = '<span class="chip {}">{}</span>'.format(owner, _html.escape(owner_text[owner])) if owner else ""
                cells.append("<td>{}</td>".format(chip))
            for pos in range(1, len(row)):
                v = row[pos]
                txt = "{:.2f}%".format(float(v)) if as_pct else str(int(v))
                p = None if is_total else pct_lookup.get(category, {}).get(cols[pos])
                cls, inner = "", '<span class="val">{}</span>'.format(_html.escape(txt))
                if p is not None and owner == "tgt":
                    cls = "ok" if float(p) >= target else "low"
                    if as_pct:
                        gap = float(p) - target
                        inner += '<div class="gap">{} {:.2f} pts vs target</div>'.format(
                            "&#9650;" if gap >= 0 else "&#9660;", abs(gap))
                elif p is not None and as_pct and owner in bar_colour and budget > 0:
                    w = min(float(p) / budget, 1.0) * 100.0
                    if float(p) > 0:
                        w = max(w, 3.0)
                    inner = '<div class="bar" style="width:{:.1f}%;background:{};"></div>'.format(
                        w, bar_colour[owner]) + inner
                cells.append('<td class="{}">{}</td>'.format(cls, inner))
            body.append('<tr class="{}">{}</tr>'.format("gt" if is_total else "", "".join(cells)))
        return '<table class="fms"><thead><tr>{}</tr></thead><tbody>{}</tbody></table>'.format(head, "".join(body))

    return (css + '<div class="fmcard"><div class="fmwrap">'
            + '<div><div class="fmtitle">Shipment Count</div>' + one(count_df, False, True) + "</div>"
            + '<div><div class="fmtitle">Percentage of DC Shipments</div>' + one(pct_df, True, False) + "</div>"
            + "</div></div>")


def refresh_output_data(out_path):
    """Re-read temporary output and rebuild DC-wise count/percentage summaries."""
    dc_counts = {}
    total_rows = 0
    for chunk in pd.read_csv(
        out_path, dtype=str, keep_default_na=False,
        usecols=lambda c: c in ["bagging_dc_name", "Final_Remarks"], chunksize=CHUNK_ROWS,
        encoding="utf-8-sig", encoding_errors="replace"
    ):
        total_rows += len(chunk)
        grouped = chunk.groupby(["bagging_dc_name", "Final_Remarks"], dropna=False).size()
        for (dc_name, remark), count in grouped.items():
            dc = str(dc_name).strip() if str(dc_name).strip() else "(Blank DC)"
            rm = str(remark).strip() if str(remark).strip() else "(Blank Remark)"
            dc_counts.setdefault(dc, {})[rm] = dc_counts.setdefault(dc, {}).get(rm, 0) + int(count)
    count_summary, pct_summary = build_dc_summaries(dc_counts)
    preview = pd.read_csv(out_path, nrows=50, dtype=str, keep_default_na=False,
                          encoding="utf-8-sig")
    return total_rows, dc_counts, count_summary, pct_summary, preview


def run_job(dash_file, map_file, processing_hours, lastveh_hours, out_path, progress=None):
    t0 = time.time()
    processing_days = processing_hours / 24.0
    lastveh_days = lastveh_hours / 24.0
    mp = read_mapping(map_file)
    hub_to_lane, lane_info = build_lane_tables(mp, processing_days)
    stats = {"rows": 0, "unmatched": {}, "bad_dates": {}, "counts": {}, "dc_counts": {}}
    total = _total_bytes(dash_file)
    dt_cols = None
    first = True
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        for ch in iter_chunks(dash_file):
            # Keep only original FM ODC columns A:CN. Ignore all columns after CN.
            ch = ch.iloc[:, :BASE_INPUT_COLS].copy()
            if dt_cols is None:
                dt_cols = detect_dt_cols(ch)
            out = process_chunk(ch, hub_to_lane, lane_info, dt_cols, processing_days, lastveh_days, stats)
            out.to_csv(fh, index=False, header=first)
            first = False
            if progress:
                try:
                    frac = min(0.99, dash_file.tell() / total) if total else None
                except Exception:
                    frac = None
                progress(stats["rows"], frac, time.time() - t0)
    stats["seconds"] = round(time.time() - t0, 1)
    stats["dt_cols"] = dt_cols or []
    return stats


def save_temp_result_to_output(source_path, output_dir, file_name):
    """Copy the temporary result into the permanent output folder only on download click."""
    import streamlit as st

    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, file_name)
    shutil.copyfile(source_path, destination)
    st.session_state["permanent_output_path"] = destination


# ---------------------------------------------------------------- UI
def main():
    import streamlit as st

    st.set_page_config(page_title="FM-to-ODC-Dashboard-48hrs-NCR", layout="wide")
    st.title("FM-to-ODC-Dashboard-48hrs-NCR")
    st.caption("Upload the two daily files, generate the report, and download the final CSV.")

    with st.sidebar:
        st.header("Settings")
        processing_hours = st.number_input("Processing Time (Hours)", value=1.0, step=0.25,
                                           help="The Excel Processing column is set to 1 hour.")
        lastveh_hours = st.number_input("Last Vehicle Received Time (Hour of Day)", value=3.0, step=0.5,
                                        help="The Excel Last_Vehicle Recieved time column is set to 03:00.")
        target_pct = st.number_input("Connected ON time Target (%)", value=CF_TARGET_PCT, step=0.5,
                                     min_value=50.0, max_value=100.0,
                                     help="Summary colours use this target (green = target met).")

    c1, c2 = st.columns(2)
    with c1:
        dash = st.file_uploader("1) FM ODC Dashboard File", type=["csv", "xlsx", "xlsm", "xlsb", "zip", "gz"])
    with c2:
        mapf = st.file_uploader("2) Lane Mapping File", type=["xlsx", "xlsm", "xlsb", "csv"])

    if st.button("Generate File", type="primary", disabled=not (dash and mapf)):
        # Remove the previous temporary result when a new job starts.
        old_result = st.session_state.get("result")
        if old_result:
            old_temp = old_result[0]
            try:
                if os.path.isfile(old_temp):
                    os.remove(old_temp)
            except OSError:
                pass
        st.session_state.pop("preview_df", None)
        st.session_state.pop("permanent_output_path", None)
        st.session_state.pop("refresh_message", None)

        # Process into the OS temporary directory. Do NOT create a permanent
        # file in the application's output folder at Generate time.
        fd, temp_path = tempfile.mkstemp(prefix="FM_ODC_work_", suffix=".csv")
        os.close(fd)
        file_stamp = time.strftime("%Y%m%d_%H%M%S")
        download_name = "FM_ODC_final_{}.csv".format(file_stamp)
        st.session_state["download_file_name"] = download_name

        st.warning("Processing is in progress. The result is being kept temporarily. "
                   "It will be saved in the output folder only when you click Download Final File (CSV).")
        bar = st.progress(0.0, text="Starting...")

        def on_progress(rows, frac, secs):
            bar.progress(frac if frac is not None else 0.0,
                         text="Processed {:,} rows ({} min {} sec)".format(rows, int(secs // 60), int(secs % 60)))

        try:
            stats = run_job(dash, mapf, processing_hours, lastveh_hours, temp_path, progress=on_progress)
            bar.empty()
            st.session_state["result"] = (temp_path, stats)
        except Exception as e:  # show a clear message and clean up a failed temporary file
            bar.empty()
            try:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            st.error("An error occurred: {}".format(e))

    res = st.session_state.get("result")
    if res:
        temp_path, stats = res
        if not os.path.exists(temp_path):
            st.error("The temporary processed data is no longer available. Please generate the file again.")
            st.stop()

        st.success("Completed! {:,} rows processed in {} seconds.".format(stats["rows"], stats["seconds"]))
        permanent_path = st.session_state.get("permanent_output_path")
        if permanent_path and os.path.exists(permanent_path):
            st.success("Permanent CSV saved in output folder: {}".format(permanent_path))
        else:
            st.info("Your processed data is currently temporary. It will be saved permanently in the output folder only when you click Download Final File (CSV).")

        if st.session_state.pop("refresh_message", False):
            st.success("Summary and preview data refreshed successfully from the temporary processed CSV.")

        action1, action2 = st.columns([1, 1])
        with action1:
            if st.button("Refresh Data"):
                try:
                    refreshed_rows, refreshed_dc_counts, count_summary, pct_summary, refreshed_preview = refresh_output_data(temp_path)
                    stats["rows"] = refreshed_rows
                    stats["dc_counts"] = refreshed_dc_counts
                    stats["count_summary"] = count_summary
                    stats["pct_summary"] = pct_summary
                    st.session_state["result"] = (temp_path, stats)
                    st.session_state["preview_df"] = refreshed_preview
                    st.session_state["refresh_message"] = True
                    st.rerun()
                except Exception as e:
                    st.error("Could not refresh the data: {}".format(e))
        with action2:
            with open(temp_path, "rb") as fh:
                st.download_button(
                    "Download Final File (CSV)", fh,
                    file_name=st.session_state.get("download_file_name", "FM_ODC_final.csv"),
                    mime="text/csv",
                    on_click=save_temp_result_to_output,
                    args=(temp_path, os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
                          st.session_state.get("download_file_name", "FM_ODC_final.csv"))
                )

        if permanent_path and os.path.exists(permanent_path):
            if st.button("Open Output Folder"):
                try:
                    os.startfile(os.path.dirname(permanent_path))
                except Exception as e:
                    st.error("Could not open the output folder: {}".format(e))

        if stats["unmatched"]:
            st.warning("The following destination_hub values were not found in the mapping file (Lane = #N/A). Please update the mapping file:")
            top = sorted(stats["unmatched"].items(), key=lambda x: -x[1])[:30]
            st.dataframe(pd.DataFrame(top, columns=["destination_hub", "rows"]), hide_index=True)
        if stats["bad_dates"]:
            st.warning("Some date-time values could not be interpreted and were treated as blank: {}".format(stats["bad_dates"]))

        st.subheader("Summary – Shipment Count & Percentage (DC-wise)")
        count_summary, pct_summary = build_dc_summaries(stats.get("dc_counts", {}))
        st.markdown(summary_tables_html(count_summary, pct_summary, target_pct), unsafe_allow_html=True)

        st.subheader("Preview (First 50 Rows)")
        preview_df = st.session_state.get("preview_df")
        if preview_df is None:
            preview_df = pd.read_csv(temp_path, nrows=50, dtype=str, keep_default_na=False,
                                     encoding="utf-8-sig")
        st.dataframe(preview_df)
        with st.expander("Date-Time Columns Cleaned"):
            st.write(stats["dt_cols"])



if __name__ == "__main__":
    main()
