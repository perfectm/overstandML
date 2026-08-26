"""Market features (spec section 4) — leakage timing is the core design rule.

Day-of (D) features, captured just after market open (~9:31 ET), just before
prediction:
    opening_vix, opening_spx, opening_gap

Prior-close (D-1) features — everything else, including VIX term structure:
    vix, vix9d, vix3m, vix6m  (VIX1D deliberately excluded so far)

Implementation note: daily-bar Open is the proxy for the ~9:31 ET capture —
close enough for research; the LIVE module should capture true 9:31 quotes.
The alignment logic (build_market_features) is separated from the data
fetch (fetch_ohlc) so the leakage rules are testable offline.
"""
from __future__ import annotations

import pandas as pd

DAY_OF_OPEN = ["opening_vix", "opening_spx", "opening_gap"]
PRIOR_CLOSE = ["vix", "vix9d", "vix3m", "vix6m"]

# The VIX family comes from CBOE's public daily-prices CSVs (Yahoo dropped
# ^VIX9D/^VIX3M/^VIX6M); SPX comes from yfinance (^GSPC).
CBOE_SYMBOLS = {"vix": "VIX", "vix9d": "VIX9D", "vix3m": "VIX3M", "vix6m": "VIX6M"}
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{symbol}_History.csv"
SPX_SYMBOL = "^GSPC"


def fetch_cboe(symbol: str) -> pd.DataFrame:
    """Full daily OHLC history for one CBOE index (DATE,OPEN,...,CLOSE)."""
    df = pd.read_csv(CBOE_URL.format(symbol=symbol), parse_dates=["DATE"])
    df = df.rename(columns={"DATE": "Date", "OPEN": "Open", "CLOSE": "Close"})
    return df.set_index("Date")[["Open", "Close"]]


def fetch_ohlc(start: str, end: str | None = None) -> dict[str, pd.DataFrame]:
    """Download daily OHLC for every feature input.
    Returns {key: DataFrame with Open/Close, DatetimeIndex}."""
    import yfinance as yf

    out = {}
    for key, symbol in CBOE_SYMBOLS.items():
        out[key] = fetch_cboe(symbol).loc[start:end]

    spx = yf.download(SPX_SYMBOL, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(spx.columns, pd.MultiIndex):  # yfinance >=0.2 single-ticker quirk
        spx.columns = spx.columns.get_level_values(0)
    out["spx"] = spx[["Open", "Close"]]
    return out


def build_market_features(ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the feature frame from {key: OHLC} per SYMBOLS.

    Row for date D holds only what is knowable at ~9:31 ET on D:
      opening_vix / opening_spx — D's Open
      opening_gap               — D's SPX Open vs D-1's SPX Close, in %
      vix, vix9d, vix3m, vix6m  — D-1's Close (shifted; NEVER D's close)
    """
    spx = ohlc["spx"]
    feats = pd.DataFrame(index=spx.index)
    feats["opening_vix"] = ohlc["vix"]["Open"]
    feats["opening_spx"] = spx["Open"]
    feats["opening_gap"] = spx["Open"] / spx["Close"].shift(1) - 1.0

    for key in PRIOR_CLOSE:
        feats[key] = ohlc[key]["Close"].reindex(feats.index).shift(1)

    # first row has no D-1 close — unusable for prediction
    return feats.iloc[1:]
