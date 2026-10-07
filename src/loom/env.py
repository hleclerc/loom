"""loom's environment variables, in ONE place.

= The prefix is `LOOM_`

It was so for the few recent settings (`LOOM_ZERO_OUTPUTS`, `LOOM_JOURNAL`) and it was not
so for all the others, which carried `SDOT_` -- the name of the package that gave birth to loom.
An outsider would install loom and receive sdot: a cache in `~/.cache/sdot`, a `SDOT_BUILD_DIR`,
error messages that say "sdot:". It is the first friction noted by
`loom/examples/diffusion`, and the only one that is fixed by a rename.

What legitimately stays at `SDOT_`: what sdot reads for itself (`SDOT_KTYPE`, the kernel type
of its cells; `SDOT_CATALOGUE_DIR`, where ITS catalogue is). The rule is the usual one here:
the prefix says who the setting belongs to.

= The old name is still read, once, saying so

These names live in scripts, containers, private `Makefile`s and shells that are not
versioned: breaking them all at once would help nobody. `SDOT_X` is therefore still read
when `LOOM_X` is absent, with a warning emitted ONLY ONCE per variable -- enough
to be seen, not enough to pollute a test suite.

= And a single place that knows how to read a switch

`flag()` carries the convention, which used to be copied verbatim into six files:
absent -> the default; `0`, `false`, `no`, `off` or empty -> false; anything else -> true.
"""
import os
import sys

PREFIX = "LOOM_"
OLD_PREFIX = "SDOT_"

_warned = set()


def _read( name ):
    """The value of `LOOM_<name>`, failing that that of `SDOT_<name>` (saying so once), otherwise
    `None`."""
    value = os.environ.get( PREFIX + name )
    if value is not None:
        return value
    value = os.environ.get( OLD_PREFIX + name )
    if value is not None and name not in _warned:
        _warned.add( name )
        print( f"loom: { OLD_PREFIX }{ name } is the old name of { PREFIX }{ name } "
               f"-- still read, please rename", file = sys.stderr )
    return value


def var( name, default = None ):
    """The setting `name`, or `default` if it is not set."""
    value = _read( name )
    return default if value is None else value


def flag( name, default = False ):
    """The setting `name` read as a switch (see the module docstring)."""
    value = _read( name )
    if value is None:
        return default
    return value.strip().lower() not in ( "", "0", "false", "no", "off" )


def is_set( name ) -> bool:
    """Whether the setting is set, under either prefix."""
    return _read( name ) is not None


def set_var( name, value ):
    """Set the setting, for ourselves and for subprocesses (what `loom-kernels` does before
    launching a recording or a catalogue compilation)."""
    os.environ[ PREFIX + name ] = str( value )


_hinted = set()


def hint( message ):
    """An optimization ADVICE, shown (once per message) only to who asked for them with `LOOM_HINTS=1`.
    loom does what it is asked even when it could be done faster -- a copy is made when one is needed,
    and this is where it says so, never by refusing."""
    if message not in _hinted and flag( "HINTS" ):
        _hinted.add( message )
        print( f"loom hint: { message }", file = sys.stderr )
