"""Data layer (spec section 3, Phase 0-1).

oo_contract  — the Option Omega column contract: the single ingestion format
ts_to_oo     — TradeSteward export -> OO contract translator (the ONLY
               TS-specific code in the system)
portfolio    — strategy CSVs -> children -> parent assembly + daily PNL
"""
