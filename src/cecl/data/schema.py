"""Freddie Mac SFLLD file layout, Release 48 (October 2026).

Field order and "Not Available" sentinel codes are transcribed from the General User
Guide, Release 48, Glossary section (data/raw/freddie/docs/user_guide.pdf). The files
have no header row, so column order is the contract: tests/test_schema.py checks the
field counts against the raw files.
"""
from __future__ import annotations

import polars as pl

S, I16, I32, F32, F64 = pl.Utf8, pl.Int16, pl.Int32, pl.Float32, pl.Float64

# (column name, dtype) in file order. 31 fields.
ORIG_FIELDS: list[tuple[str, pl.DataType]] = [
    ("fico", I16),
    ("first_pay_date", I32),          # YYYYMM
    ("first_time_buyer", S),          # Y / N / 9
    ("maturity_date", I32),           # YYYYMM
    ("msa", S),
    ("mi_pct", I16),
    ("units", I16),
    ("occupancy", S),                 # P primary, I investment, S second home, 9 NA
    ("cltv", I16),
    ("dti", I16),
    ("orig_upb", F64),
    ("ltv", I16),
    ("orig_rate", F32),
    ("channel", S),                   # R retail, B broker, C correspondent, T TPO NS, 9 NA
    ("ppm_flag", S),
    ("amort_type", S),                # FRM / ARM
    ("state", S),
    ("prop_type", S),                 # SF, CO, PU, MH, CP, 99
    ("zip3", S),
    ("loan_id", S),
    ("purpose", S),                   # P purchase, C cash-out, N no cash-out, R refi NS, 9 NA
    ("orig_term", I16),
    ("n_borrowers", I16),
    ("seller_name", S),
    ("super_conforming", S),
    ("pre_harp_loan_id", S),
    ("program", S),
    ("harp_flag", S),
    ("valuation_method", S),
    ("io_flag", S),
    ("vantage4", I16),
]

# (column name, dtype) in file order. 35 fields.
PERF_FIELDS: list[tuple[str, pl.DataType]] = [
    ("loan_id", S),
    ("period", I32),                  # YYYYMM reporting month
    ("cur_upb", F64),
    ("delq", S),                      # 00,01,02,... months delinquent; RA = REO acquired; XX = NA
    ("loan_age", I16),
    ("rem_months", I16),
    ("defect_settle_date", I32),
    ("mod_flag", S),
    ("zb_code", S),
    ("zb_date", I32),
    ("cur_rate", F32),
    ("cur_nonint_upb", F64),          # deferred (non-interest-bearing) UPB
    ("ddlpi", I32),                   # due date of last paid installment
    # Loss fields: populated at disposition. Empirically, recoveries are stored as
    # negative numbers, expenses positive, Actual Loss positive (see tests/test_loss_identity).
    ("mi_recoveries", F64),
    ("net_sales_proceeds", F64),
    ("non_mi_recoveries", F64),
    ("total_expenses", F64),
    ("legal_costs", F64),
    ("maint_pres_costs", F64),
    ("taxes_ins", F64),
    ("misc_expenses", F64),
    ("actual_loss", F64),
    ("cum_mod_cost", F64),
    ("step_mod_flag", S),
    ("payment_deferral", S),
    ("eltv", I16),
    ("zb_removal_upb", F64),
    ("delinq_accrued_int", F64),
    ("disaster_flag", S),
    ("borrower_assist", S),           # F forbearance, R repayment, T trial mod, etc.
    ("cur_period_mod_cost", F64),
    ("cur_int_upb", F64),
    ("mi_cancel", S),
    ("servicer_name", S),
    ("bk_cramdown", F64),
]

# Codes meaning "Not Available" per field (User Guide R48). Replaced by null on cleaning.
ORIG_SENTINELS: dict[str, list] = {
    "fico": [9999], "vantage4": [9999], "first_time_buyer": ["9"], "mi_pct": [999],
    "units": [99], "occupancy": ["9"], "cltv": [999], "dti": [999], "ltv": [999],
    "channel": ["9"], "prop_type": ["99"], "purpose": ["9"], "n_borrowers": [99],
    "valuation_method": ["9"],
}
PERF_SENTINELS: dict[str, list] = {"eltv": [999], "delq": ["XX"]}


def raw_schema(fields: list[tuple[str, pl.DataType]]) -> dict[str, pl.DataType]:
    """Read every column as text first; typed casting happens in one audited step."""
    return {name: pl.Utf8 for name, _ in fields}


def typed(fields: list[tuple[str, pl.DataType]]) -> list[pl.Expr]:
    return [pl.col(n).str.strip_chars().cast(t, strict=True).alias(n) for n, t in fields]
