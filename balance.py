"""Your Claude API balance, for Flash's status line while a @claude model
is in use (on in this extension's settings, off by default).

The API has no way to read the prepaid credit left on an account, so this
reads what it can: what the organization has spent, from the Usage and
Cost Analytics API, which needs an Admin API key (Claude Console,
Settings, Admin keys). Given the balance you had on a day (the two
"Balance" settings), it says roughly what is left: that balance less what
was spent since. Without one, it says what this month has cost so far.
The figures run a few hours behind (the API refreshes within 4 to 24
hours), so the line says how fresh they are.

Prints one line, or nothing when turned off.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(os.environ.get("FLASH_EXTENSION_DIR") or Path(__file__).parent)
sys.path.insert(0, str(HERE))

KEPT = HERE / ".balance.json"
KEEP_FOR = 15 * 60          # the data changes a few times a day at most
LONGEST = dt.timedelta(days=31)   # what one request may span
EARLIEST = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def setting(name: str) -> str:
    return os.environ.get(f"FLASH_SETTING_{name.upper()}", "").strip()


def admin_key() -> str:
    return setting("admin_key") or os.environ.get("ANTHROPIC_ADMIN_KEY", "")


def since() -> tuple[dt.datetime, float | None]:
    """Where to count from, and the balance then: the day the balance
    setting is for, or the start of this month when there is none."""

    now = dt.datetime.now(dt.timezone.utc)
    try:
        balance = float(setting("balance") or 0) or None
    except ValueError:
        balance = None
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if balance is not None:
        try:
            day = dt.date.fromisoformat(setting("balance_date"))
            start = dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)
        except ValueError:
            pass
    start = max(start, EARLIEST, now - dt.timedelta(days=364))
    return start, balance


def spent(client, start: dt.datetime) -> tuple[float, str | None]:
    """Dollars spent from START until now, and when the data was last
    refreshed. Asked 31 days at a time, the most one request covers."""

    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    cents = 0.0
    refreshed = None
    at = start
    while at < now:
        end = min(at + LONGEST, now)
        page = client.beta.organization.analytics.cost.list(
            starting_at=at.isoformat().replace("+00:00", "Z"),
            ending_at=end.isoformat().replace("+00:00", "Z"),
        )
        for row in page:
            cents += float(row.cost.amount or 0)
        refreshed = getattr(page, "data_refreshed_at", None) or refreshed
        at = end
    return cents / 100, (str(refreshed) if refreshed else None)


def fresh_note(refreshed: str | None) -> str:
    if not refreshed:
        return ""
    try:
        when = dt.datetime.fromisoformat(refreshed.replace("Z", "+00:00"))
    except ValueError:
        return ""
    hours = (dt.datetime.now(dt.timezone.utc) - when).total_seconds() / 3600
    return " · now" if hours < 1 else f" · {int(hours)}h ago"


def line() -> str:
    if setting("show_balance") != "1":
        return ""
    key = admin_key()
    if not key:
        return "Claude: add an Admin API key to see your balance"
    start, balance = since()
    marker = f"{start.date()}|{balance}"
    try:
        kept = json.loads(KEPT.read_text(encoding="utf-8"))
        if kept.get("for") == marker and time.time() - kept.get("at", 0) < KEEP_FOR:
            return kept["line"]
    except (OSError, ValueError, KeyError):
        pass

    from provider import sdk

    anthropic = sdk()
    client = anthropic.Anthropic(api_key=key)
    if not hasattr(getattr(client.beta, "organization", None), "analytics"):
        # An older copy of the package, from before the analytics API: it
        # updates, and the next look (a fresh program) has the new one.
        from provider import DEPS, _install

        try:
            _install("anthropic", DEPS, upgrade=True)
        except RuntimeError:
            return "Claude: update the anthropic package to see your balance"
        return "Claude: balance shows in a few minutes"
    try:
        dollars, refreshed = spent(client, start)
    except anthropic.AuthenticationError:
        return "Claude: that Admin API key isn't accepted"
    except anthropic.PermissionDeniedError:
        return "Claude: the balance needs an Admin API key"
    except anthropic.APIError:
        return ""
    when = fresh_note(refreshed)
    if balance is None:
        said = f"Claude ${dollars:,.2f} this month{when}"
    else:
        said = f"Claude ≈ ${max(balance - dollars, 0):,.2f} left{when}"
    try:
        KEPT.write_text(json.dumps({"for": marker, "at": time.time(), "line": said}), encoding="utf-8")
    except OSError:
        pass
    return said


if __name__ == "__main__":
    print(line())
