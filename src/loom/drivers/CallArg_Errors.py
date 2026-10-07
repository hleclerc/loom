import numpy

from .IoCategory import IoCategory
from .CallArg import CallArg


# the C++ name of the buffer in the generated handler: what a value that can fail is handed.
ERRORS_VAR_NAME = "errors"

# how many records the buffer holds. A record past the last is DROPPED (see ErrorBuffer.h): the
# host still learns that something failed, and since it doubles anyway, a dropped record costs at
# most one more run.
MAX_RECORDS = 8

# error kinds, as C++ writes them (support/containers/ErrorBuffer.h)
CAPACITY_OVERFLOW = 1
FAILURE           = 2


class KernelFailure( RuntimeError ):
    """A kernel said that something went wrong that running it again would not fix (an
    `ErrorKind::failure` record): the message is the one the call gave for that code."""


def failure_message( call_name, records, messages ):
    """The text of the failure records `[ ( kind, id, value ) ]` -- `None` if there are none. A record
    of a kind the host does not know is reported too: silently returning what a kernel said was wrong
    is the one thing not to do."""
    lines = []
    for kind, id, value in records:
        if kind == CAPACITY_OVERFLOW:
            continue
        if kind == FAILURE and id in messages:
            lines.append( messages[ id ].format( value = value ) )
        else:
            lines.append( f"error kind { kind }, code { id }, value { value }" )
    if not lines:
        return None
    return f"{ call_name }: " + "; ".join( dict.fromkeys( lines ) )


class CallArg_Errors( CallArg ):
    """The error buffer of a call: where the kernel says that something went wrong.

    One per call, shared by everything that can fail -- a capacity that did not fit, today; a
    degenerate cell or a division by zero tomorrow. Nothing about it is ShapeVar business: the
    values that can fail merely hold a view on it (see `ErrorBuffer.h`, and why "a global" is not
    an option on a device).

    It is an OUTPUT buffer like any other, but not one of the caller's: nothing is written back
    onto a Python object. What we do with it is decide whether the call has to be run again --
    with more room this time (`JaxDriver.call`).
    """

    def __init__( self, call_args_analysis ) -> None:
        super().__init__( IoCategory.OUTPUT, ERRORS_VAR_NAME )

        self.memory_space = call_args_analysis.cpp_memory_space
        self.max_records = MAX_RECORDS
        self.shape = [ 1 + 3 * self.max_records ]   # [ nb_records, ( kind, id, value )... ]
        self.raw = None
        # what the call says a failure code means ( `loom.ffi_call( ..., failures = { code: message } )` ),
        # and its name, for the message
        self.failure_messages = {}
        self.call_name = getattr( call_args_analysis, "name", "" ) or ""

    def is_ffi_buffer( self ):
        return True

    def takes_batch_axis( self ):
        """ONE buffer for the whole call, batched or not: every item of a batch records into this
        very one (atomically). So a `vmap` gives it no axis -- and it comes back unbatched."""
        return False

    # -- driver-agnostic C++ (the same for every driver) --
    def cpp_root_decl( self, var_name ):
        view = ( f"tensor_view<{ self.memory_space }>( { self.jax_data_ptr() }, "
                 f"tuple( { self.jax_dim( 0 ) } ) )" )
        return f"    auto { var_name } = make_error_buffer( { view }, SI( { self.max_records } ) );"

    def cpp_seed_root( self, var_name ):
        # no record, no error: the count of records is what says whether anything failed, so it
        # has to start at zero -- through the queue, like any other device buffer.
        return f"    { var_name }.fill_with( queue, 0 );"

    # -- what the host makes of it --
    def capacity_overflows( self, error_vars ):
        """The counts that did not fit, as `[ ( path, wanted, capacity ) ]` -- one entry per
        offending ShapeVar, `wanted` being the largest count it asked for.

        `None` when the buffer is a TRACED value (under a `jit` or a `vmap`): its content only
        exists at execution time, so no Python loop can look at it and try again."""
        try:
            from ..tensor.host import to_host
            raw = to_host( self.raw )
        except Exception:
            return None

        nb_records = min( int( raw[ 0 ] ), self.max_records )
        records = [ tuple( int( v ) for v in raw[ 1 + 3 * num : 4 + 3 * num ] ) for num in range( nb_records ) ]

        # a FAILURE is not a capacity: running again would not fix it, so it is raised here
        message = failure_message( self.call_name, records, self.failure_messages )
        if message is not None:
            raise KernelFailure( message )

        wanted_of = {}
        for kind, id, value in records:
            if kind == CAPACITY_OVERFLOW:
                wanted_of[ id ] = max( wanted_of.get( id, 0 ), value )

        return [ ( error_vars[ id ].path, wanted, error_vars[ id ].max_bound )
                 for id, wanted in wanted_of.items() ]

    # -- Jax FFI ABI --
    def jax_ffi_type( self ):
        return "ffi::BufferR1<ffi::S32>"

    def out_shape_dtype( self ):
        import numpy
        return tuple( self.shape ), numpy.dtype( numpy.int32 )

    def jax_out_spec( self ):
        import jax
        import jax.numpy as jnp
        return jax.ShapeDtypeStruct( tuple( self.shape ), jnp.int32 )

    def jax_write_back( self, array ):
        # nothing to write back onto a Python object: this one is for us.
        self.raw = array
