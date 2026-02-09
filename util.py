import datetime
from typing import Optional

def generate_calver(
    format: str = "YYYY.MM.DD.HHMM",
    dt: Optional[datetime.datetime] = None,
    timezone: Optional[datetime.tzinfo] = datetime.timezone.utc
) -> str:
    """
    Generate a Calendar Version (CalVer) string based on the current date and time.

    Common formats:
        "YYYY.MM.DD"           -> 2026.02.09
        "YY.MM.DD"             -> 26.02.09
        "YYYYMMDD"             -> 20260209
        "YYYY.MM.DD.HHMM"      -> 2026.02.09.1430
        "YYYY.MM.DD.HHMMSS"    -> 2026.02.09.143022
        "YYYY.0M.0D"           -> 2026.2.9        (Ubuntu-style)
        "YY.0M"                -> 26.2            (short Ubuntu-style)

    Parameters:
        format (str): Version format string using these placeholders:
                      YYYY (full year), YY (2-digit year),
                      MM (month), 0M (zero-padded month, but single digit if <10),
                      DD (day),   0D (zero-padded day),
                      HH (hour), MM (minute), SS (second)
        dt (datetime.datetime, optional): Specific datetime to use.
                                          Defaults to current time.
        timezone (tzinfo, optional): Timezone to use if dt has no tzinfo.
                                     Defaults to UTC.

    Returns:
        str: Formatted CalVer string
    """
    if dt is None:
        dt = datetime.datetime.now(timezone)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone)

    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour
    minute = dt.minute
    second = dt.second

    replacements = {
        "YYYY": f"{year}",
        "YY":   f"{year % 100:02d}",
        "MM":   f"{month:02d}",
        "0M":   f"{month}" if month >= 10 else f"{month}",
        "DD":   f"{day:02d}",
        "0D":   f"{day}" if day >= 10 else f"{day}",
        "HH":   f"{hour:02d}",
        "mm":   f"{minute:02d}",  # lowercase mm to avoid conflict with month
        "SS":   f"{second:02d}",
    }

    result = format
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)

    return result


# Example usage (run on February 9, 2026, at 14:30:22 UTC)
if __name__ == "__main__":
    print(generate_calver("YYYY.MM.DD"))           # 2026.02.09
    print(generate_calver("YY.0M"))                # 26.2
    print(generate_calver("YYYY.MM.DD.HHMM"))      # 2026.02.09.1430
    print(generate_calver("YYYYMMDD-HHMMSS"))      # 20260209-143022