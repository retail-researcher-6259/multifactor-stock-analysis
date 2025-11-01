# Options Flow Factor - Diagnostic Report

## Issue Summary

The `options_flow` factor is returning constant values of **0.5** for all stocks, indicating the factor is not working correctly.

## **STATUS: DISABLED** ❌

As of 2025-11-01, the OptionsFlow factor has been **disabled** in MultiFactor_optimizer_07.py due to Yahoo Finance data access issues. The factor now returns a constant 0.5 (neutral) and has weight set to 0.0.

## Root Cause Analysis

### 1. **Default Return Value**
The `get_options_flow_score()` function returns `0.5` (neutral) in three scenarios:

- **Line 1368**: Insufficient historical data (< 2 days cached)
  ```python
  if len(historical_oi) < 2:
      return 0.5  # Neutral score
  ```

- **Line 1426**: Minimal options activity detected
  ```python
  if total_activity < 0.1:
      return 0.5  # Neutral
  ```

- **Line 1436**: Exception during processing
  ```python
  except Exception as e:
      return 0.5  # Neutral on error
  ```

### 2. **Most Likely Cause: No Historical Data**

Since this is the first run, there is **no cached historical options data**. The function requires:

- At least **2 days** of cached options data
- Data stored in `./cache/options_oi/` directory
- Format: `{ticker}_{YYYY-MM-DD}.json`

**On the first run**, the cache is empty, so the function **always returns 0.5**.

### 3. **Additional Potential Issues**

#### a) Options Data Fetching Failures
```python
# Line 1355-1361
try:
    t = Ticker(ticker, session=_global_curl_session)
    options_chain = t.option_chain
    cache_options_oi(ticker, options_chain)
except Exception as e:
    print(f"  ⚠️ Could not fetch options for {ticker}: {e}")
```

**Problem**: Exceptions are caught and suppressed. If options fetching fails, you won't see errors.

#### b) Not All Stocks Have Options
- Only large-cap, liquid stocks have active options markets
- Small-cap stocks may have no options or very limited options activity
- ETFs typically have more liquid options than individual stocks

#### c) Data Provider Limitations
- Yahoo Finance options data may be limited or delayed
- Some tickers may require authentication
- Rate limiting may apply

## How the Options Flow Factor Works

### Data Collection Process
1. **Fetch options chain** from Yahoo Finance for each ticker
2. **Cache daily snapshots** of open interest (OI) data
3. **Track OI changes** over time (14-day lookback by default)
4. **Filter significant flows**:
   - OI change > 100 contracts
   - Notional value > $500,000
5. **Apply time decay** (5-day half-life)
6. **Calculate sentiment**:
   - Call increases = Bullish signal
   - Put increases = Bearish signal

### Score Calculation
- **Score range**: 0.0 to 1.0
- **0.5** = Neutral (no significant flow)
- **> 0.5** = Bullish (net call buying)
- **< 0.5** = Bearish (net put buying)

## Solutions & Recommendations

### Short-Term: Remove or Disable Options Flow

Since the factor requires multiple days of data collection:

**Option 1**: Set weight to 0 (already done in your test)
```python
WEIGHTS = {
    "options_flow": 0.0,  # Disabled until cache builds up
    # ... other factors
}
```

**Option 2**: Comment out the factor calculation
```python
# options_flow = get_options_flow_score(tk, info)
options_flow = 0.5  # Placeholder
```

### Medium-Term: Build Historical Cache

Run the script daily for **2-3 weeks** to build cache:

1. **Day 1**: Fetches today's options data, returns 0.5 (no historical data)
2. **Day 2**: Has 2 days of data, can calculate momentum
3. **Day 3+**: More accurate with longer history

**Recommended approach**:
```bash
# Run daily as a cron job
0 16 * * 1-5 cd /path/to/weight_optimization && python MultiFactor_optimizer_07.py
```

### Long-Term: Improve Implementation

#### 1. Add Verbose Logging
```python
def get_options_flow_score(ticker, info, lookback_days=14, decay_halflife=5, verbose=False):
    # ... existing code ...

    if verbose:
        print(f"  📊 {ticker}: {len(historical_oi)} days cached")
        if len(historical_oi) >= 2:
            print(f"     Call momentum: {call_momentum:.2f}")
            print(f"     Put momentum: {put_momentum:.2f}")
            print(f"     Score: {normalized:.3f}")
```

#### 2. Fallback to Daily Download
If cache is empty, download historical options data from a data provider:
- CBOE Options Data
- OPRA (Options Price Reporting Authority)
- Commercial providers (Polygon.io, IEX Cloud, etc.)

#### 3. Alternative: Use Options Volume Instead of OI
Instead of tracking OI changes (requires history), use current-day metrics:
- Put/Call ratio
- Unusual options activity (UOA)
- Options volume vs. average

## Testing Procedure

### Step 1: Run the Test Script
```bash
cd /home/user/multifactor-stock-analysis/src/weight_optimization
python test_options_fetching.py
```

This will:
- Test options data fetching for 5 liquid tickers
- Show detailed diagnostic information
- Build initial cache files

### Step 2: Check Cache Directory
```bash
ls -la ./cache/options_oi/
```

Expected output:
```
AAPL_2025-11-01.json
MSFT_2025-11-01.json
SPY_2025-11-01.json
...
```

### Step 3: Re-run Tomorrow
After 24 hours, run the optimizer again. The options flow should start working.

### Step 4: Monitor Results
Look for:
- Varying options_flow scores (not all 0.5)
- Correlation with returns
- Diagnostic messages in output

## Expected Behavior After Cache Builds Up

After 2-3 days of running:

```
Factor Correlations with Returns:
          Factor   Pearson  p-value  Spearman Significant
financial_health  0.398144 0.082100  0.594721         Yes
       technical  0.454833 0.043913  0.571429         Yes
        momentum  0.621511 0.003442  0.507531         Yes
    options_flow  0.234567 0.123456  0.345678          No  <-- Should vary!
         ...
```

## When to Exclude Options Flow

Consider excluding this factor if:

1. **Testing with small-cap stocks** (no options available)
2. **International markets** (Yahoo data may be limited)
3. **Short timeframes** (< 2 weeks of cache)
4. **Real-time analysis required** (can't wait for cache to build)

## Conclusion

**Current Status**: ❌ **DISABLED** - Yahoo Finance options data not accessible.

**Test Results** (2025-11-01):
- Tested 5 liquid tickers: AAPL, MSFT, SPY, TSLA, NVDA
- yahooquery returned 14 expiration dates but **0 contracts** for all tickers
- yfinance installation failed due to dependency conflicts
- **Decision**: Disable OptionsFlow factor until data access is resolved

**Implementation Changes**:
1. Line 2134: `options_flow = 0.5  # Disabled - data access issues`
2. Line 117: `"options_flow": 0.0  # DISABLED: Yahoo Finance options data not accessible`
3. Factor effectively removed from analysis (weight = 0, constant value = 0.5)

**Future Considerations**:
- Try alternative data providers (CBOE, OPRA, commercial APIs)
- Consider simpler options metrics (put/call ratio, implied volatility)
- Monitor Yahoo Finance API changes for potential fixes

---

**Date**: 2025-11-01
**Version**: MultiFactor_optimizer_07.py
**Status**: OptionsFlow factor DISABLED
