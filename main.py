#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase, one case at a time.

Reads every <id>.in in the input directory (MAKEFUN_INPUT_DIR, else /input)
and writes <id>.out to the output directory (MAKEFUN_OUTPUT_DIR, else
/output): for each of the game's trophies, five quantiles of its earn rate
(percent of tracked players holding it) at the case's target, at the levels
0.05, 0.25, 0.5, 0.75 and 0.95.

The model, per trophy, works in logit space on rates clamped to
[0.01, 99.99] and blends three views of the same history:

  * persistence: the rate in the case's last read;
  * a trend: a least-squares line of logit(rate) against ln(1 + hours since
    the game's release), fitted on the recent reads and read at the target;
  * a counts view: earners (rate x players tracked) and players tracked each
    fitted as a line of their log against ln(1 + hours), then divided at the
    target, so a rate that falls only because new players joined is read as
    dilution, not as players giving the trophy up.

The median is the middle of the three views' logits. The spread starts from
a random-walk width that grows with the square root of the horizon in days,
then widens with the trend fit's residual, with the size of the jumps the
history actually shows between reads, and with how far the trend is being
pushed past the last read.

One special case: a history in which no read ever moved (every rate and the
tracked count frozen) means the tracker page is stale and a refresh is due.
Then the median is shifted down a little -- a refresh dilutes rates, because
newly tracked players hold fewer trophies -- and the spread is widened a
lot, so the band can cover the refresh landing either side of the median.

Standard library only, no network, deterministic: the same case always
writes the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Random-walk width, in logit units per square-root day.
WALK = 0.5
# A trend line is never allowed to drag the median this far from the last
# rate, in logit units: one noisy slope must not throw the forecast away.
MAX_TREND_MOVE = 1.2
# The spread is kept inside this band, in logit units.
SPREAD_MIN = 0.45
SPREAD_MAX = 2.0
# How far a stale page's refresh is expected to drag rates down, in logit
# units, when a refresh is certain; scaled by the chance one lands.
REFRESH_DRAG = 0.35
# How uncertain that drag is, in logit units: a refresh can dilute a
# trophy a little or a lot, depending on how many new players it brings.
REFRESH_DRAG_SPREAD = 0.30
# Typical gap between tracker refreshes, in hours, for that chance.
REFRESH_GAP_HOURS = 8.0


def clamp_rate(pct):
    """A rate in percent, kept where its logit is defined."""
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    return round(x * 10000.0) / 10000.0


def parse_time(text):
    """An ISO 8601 timestamp to an aware datetime; None when unparsable."""
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def hours_since_release(when):
    if when is None:
        return 0.0
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def log_time(hours):
    return math.log1p(hours)


def least_squares(xs, ys, ws=None):
    """Slope and intercept of y on x, optionally weighted.

    Returns (slope, intercept, residual_std). With fewer than two distinct
    x values, or a degenerate fit, returns a flat line at the (weighted)
    mean of y with zero residual.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if ws is None:
        ws = [1.0] * n
    wsum = sum(ws)
    if wsum <= 0:
        return 0.0, 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / wsum
    my = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = math.sqrt(sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / wsum)
        return 0.0, my, resid
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = math.sqrt(
        sum(w * (y - (intercept + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys))
        / wsum
    )
    return slope, intercept, resid


def median3(a, b, c):
    return sorted((a, b, c))[1]


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(level, mid, spread, shift, shift_spread, weight):
    """The `level` quantile of a two-part mixture, in logit space.

    With probability 1 - weight the logit is Normal(mid, spread); with
    probability weight it is Normal(mid - shift, hypot(spread,
    shift_spread)). Found by bisection on the mixture's cumulative
    distribution, so the quantiles are honest about a maybe-event: a small
    chance of a downward jump pulls the low quantiles down while leaving
    the median and the high quantiles near where they were.
    """
    lo, hi = mid - 8.0, mid + 8.0
    for _ in range(60):
        m = 0.5 * (lo + hi)
        cdf = (1.0 - weight) * _norm_cdf((m - mid) / spread) + weight * _norm_cdf(
            (m - (mid - shift)) / math.hypot(spread, shift_spread)
        )
        if cdf < level:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


def trophy_forecast(rates, players, read_times, horizon_minutes, target_time):
    """Five quantiles for one trophy.

    rates: the trophy's rate per read, oldest first (None where missing).
    players: players tracked per read. read_times: each read's datetime.
    """
    horizon_days = horizon_minutes / 1440.0
    horizon_hours = horizon_minutes / 60.0

    # Only reads that actually list this trophy.
    points = [
        (read_times[j], rates[j], players[j])
        for j in range(len(rates))
        if rates[j] is not None and players[j]
    ]
    if not points:
        # No history at all for this trophy: a wide, neutral band.
        mid = logit(1.0)
        spread = SPREAD_MAX
        return [round4(min(99.99, max(0.01, inv_logit(mid + z * spread)))) for z in Z]

    last_rate = points[-1][1]
    last_logit = logit(last_rate)

    # The 100% trophy: every tracked player holds it (the opening story
    # beat), and the page's top rate cannot climb further. The scorer
    # clamps 100 to 99.99 in logit terms, so pin it there with a narrow
    # band: on the logit scale a loose band at the top is ruinously
    # expensive if the rate stays at 100, which is the likely case.
    if last_rate >= 99.99:
        mid = logit(99.99)
        spread = 0.20
        return [round4(min(99.99, max(0.01, inv_logit(mid + z * spread)))) for z in Z]

    xs = [log_time(hours_since_release(t)) for t, _, _ in points]
    logits = [logit(r) for _, r, _ in points]
    x_last = xs[-1]
    x_target = log_time(hours_since_release(target_time))
    x_span = max(0.0, x_target - x_last)

    # --- View 1: persistence is just last_logit. ---

    # --- View 2: logit trend on the recent reads. ---
    # Keep roughly the last day of reads; with fewer, use them all. Weight
    # recent reads more so an old plateau cannot pin the slope.
    recent = [(x, y) for x, y in zip(xs, logits) if x >= x_last - math.log1p(24.0)]
    if len(recent) < 3:
        recent = list(zip(xs, logits))
    rxs = [x for x, _ in recent]
    rys = [y for _, y in recent]
    rws = [1.0 + 2.0 * (x - rxs[0]) / max(1e-9, rxs[-1] - rxs[0]) for x in rxs]
    slope, intercept, resid = least_squares(rxs, rys, rws)
    trend_logit = intercept + slope * x_target
    # Cap how far the trend may move the median from the last rate.
    trend_logit = min(last_logit + MAX_TREND_MOVE,
                      max(last_logit - MAX_TREND_MOVE, trend_logit))

    # --- View 3: earners and players tracked, each in log time. ---
    earners = [max(1.0, r / 100.0 * n) for _, r, n in points]
    tracked = [float(n) for _, _, n in points]
    log_e = [math.log(e) for e in earners]
    log_n = [math.log(n) for n in tracked]
    e_slope, e_int, _ = least_squares(xs, log_e)
    n_slope, n_int, _ = least_squares(xs, log_n)
    e_target = math.exp(e_int + e_slope * x_target)
    n_target = math.exp(n_int + n_slope * x_target)
    counts_rate = 100.0 * e_target / max(1.0, n_target)
    counts_logit = logit(counts_rate)
    counts_logit = min(last_logit + MAX_TREND_MOVE,
                       max(last_logit - MAX_TREND_MOVE, counts_logit))

    mid = median3(last_logit, trend_logit, counts_logit)

    # --- Spread. ---
    walk = WALK * math.sqrt(horizon_days)
    # How much the history actually jumped between reads, in logit units:
    # refreshes land as steps, and the target band has to cover steps of
    # that size. Use the median step so one wild read cannot dominate.
    steps = [abs(logits[j] - logits[j - 1]) for j in range(1, len(logits))]
    jump = sorted(steps)[len(steps) // 2] if steps else 0.0
    # Uncertainty in pushing the trend past the last read.
    trend_unc = abs(slope) * x_span * 0.5
    spread = math.sqrt(walk ** 2 + resid ** 2 + (0.6 * jump) ** 2 + trend_unc ** 2)

    # --- Stale page: no read ever moved. ---
    all_flat = all(r == points[0][1] for _, r, _ in points)
    same_base = all(n == points[0][2] for _, _, n in points)
    if all_flat and same_base and len(points) >= 2:
        # The page has not refreshed since the first read, so one is due:
        # its chance grows with the horizon, and when it lands the newly
        # tracked players dilute every rate below today's. Score the two
        # possibilities as a mixture instead of one wide band: the low
        # quantiles fall toward the refreshed value while the median and
        # the high quantiles stay near today's rate.
        p_refresh = 1.0 - math.exp(-horizon_hours / REFRESH_GAP_HOURS)
        row = []
        for tau in LEVELS:
            q = mixture_quantile(tau, mid, walk, REFRESH_DRAG,
                                 REFRESH_DRAG_SPREAD, p_refresh)
            row.append(round4(min(99.99, max(0.01, inv_logit(q)))))
        for j in range(1, len(row)):
            row[j] = max(row[j], row[j - 1])
        return row

    spread = min(SPREAD_MAX, max(SPREAD_MIN, spread))
    row = []
    for z in Z:
        q = inv_logit(mid + z * spread)
        row.append(round4(min(99.99, max(0.01, q))))
    # Rounding cannot break the order (z rises), but make it certain.
    for j in range(1, len(row)):
        row[j] = max(row[j], row[j - 1])
    return row


def forecast_case(case):
    """The forecast object for one parsed case."""
    trophies = case.get("trophies", [])
    history = case.get("history", [])
    horizon = float(case.get("horizon_minutes", 0) or 0)
    target_time = parse_time(case.get("target"))
    if target_time is None:
        target_time = parse_time(case.get("at"))

    read_times = [parse_time(h.get("at")) for h in history]
    players = [h.get("players_tracked") for h in history]
    rows = []
    for idx, trophy in enumerate(trophies):
        rates = [
            (h.get("rates") or [None] * len(trophies))[idx]
            if idx < len(h.get("rates") or []) else None
            for h in history
        ]
        rows.append({
            "id": trophy["id"],
            "q": trophy_forecast(rates, players, read_times, horizon, target_time),
        })
    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise SystemExit(f"{case_id}: forecast over the 65,536 byte cap")
        with open(os.path.join(OUTPUT_DIR, f"{case_id}.out"), "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()
