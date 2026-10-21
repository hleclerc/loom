#!/usr/bin/env python3
"""The version of loom-ffi:  python scripts/version.py [next | VERSION]

Without argument, print it. `next`: `year.month.day.number`, the release made today ( `number` counts the releases of the day:
1, then 2, ... ). No padding: PEP 440 would drop it, `2026.10.5.1` is what pip shows for the 5th. sdot requires at least the
version of the loom it is built with ( its `scripts/wheels.py --version` writes it ), so loom is released first.
"""
from pathlib import Path
import datetime
import re
import sys

TOML = Path( __file__ ).resolve().parents[ 1 ] / "pyproject.toml"


def current():
    return re.search( r'(?m)^version = "([^"]*)"', TOML.read_text() ).group( 1 )


def next_version( version, today = None ):
    today = today or datetime.date.today()
    stem = f"{ today.year }.{ today.month }.{ today.day }"
    m = re.fullmatch( re.escape( stem ) + r"\.(\d+)", version )
    return f"{ stem }.{ int( m.group( 1 ) ) + 1 if m else 1 }"


if __name__ == "__main__":
    if len( sys.argv ) > 1:
        version = next_version( current() ) if sys.argv[ 1 ] == "next" else sys.argv[ 1 ]
        TOML.write_text( re.sub( r'(?m)^version = "[^"]*"', f'version = "{ version }"', TOML.read_text(), count = 1 ) )
    print( current() )
