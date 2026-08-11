from .collect_attributes import collect_attributes_instance
from devtools import debug, sformat
import sys


def _is_array( obj ):
    return hasattr( obj, 'shape' ) and hasattr( obj, 'dtype' ) and not isinstance( obj, type )


def _array_to_str( obj, nl ):
    import numpy
    try:
        arr = numpy.asarray( obj )
    except Exception:
        return str( obj )

    if arr.ndim == 0:
        return repr( arr.item() )

    header = f"shape={list( arr.shape )} {arr.dtype}"

    if arr.ndim > 2:
        return header + nl + str( arr )

    flat = arr if arr.ndim == 2 else arr[ numpy.newaxis, : ]

    def fmt( v ):
        if numpy.issubdtype( arr.dtype, numpy.floating ):
            return f"{ float( v ):.6g}"
        if numpy.issubdtype( arr.dtype, numpy.integer ):
            return str( int( v ) )
        return str( v )

    cells  = [ [ fmt( v ) for v in row ] for row in flat ]
    widths = [ max( [ len( cells[ r ][ c ] ) for r in range( len( cells ) ) ] + [ 0 ] ) for c in range( flat.shape[ 1 ] ) ]
    lines  = [ "  ".join( cell.rjust( widths[ c ] ) for c, cell in enumerate( row ) ) for row in cells ]
    body   = nl.join( lines )

    if arr.ndim == 1:
        return body
    return header + nl + body


def custom_printer_impl( obj, nl ):
    if isinstance( obj, ( list, tuple ) ):
        if len( obj ) == 0:
            if isinstance( obj, tuple ):
               return "()"
            return "[]"
        nnl = nl + "  "
        res = ""
        for v in obj:
            res += nnl + custom_printer_impl( v, nnl )
        return res

    if type( obj ).__repr__ is not object.__repr__:
        repr = obj.__repr__().split( "\n" )
        if len( repr ) >= 2:
            res = repr[ 0 ]
            for r in repr[ 1: ]:
                res += nl + "  " + r.rstrip()
            return res
        return '\n'.join( repr )

    if type( obj ).__str__ is not object.__str__:
        return obj.__str__()

    if _is_array( obj ):
        return _array_to_str( obj, nl + "  " )

    if isinstance( obj, dict ):
        nnl = nl + "  "
        res = ""
        for n, v in obj.items():
            res += nnl + str( n ) + ": " + custom_printer_impl( v, nnl )
        return res

    if hasattr( obj, "__dict__" ):
        res = type( obj ).__name__
        nnl = nl + "  "
        for n, _ in collect_attributes_instance( obj, use_annotations = True ):
            res += nnl + n + ": " + custom_printer_impl( getattr( obj, n, None ), nnl )
        return str(res)

    return str( obj )


def info( *arg, frame_depth = 2, **kwargs ):
    f = debug._process( arg, kwargs, frame_depth = frame_depth )
    s = sformat( f"{ f.filename }:{ f.lineno }:", sformat.blue )
    for argument in f.arguments:
        if s:
            s += "\n  "
        else:
            s += "  "
        s += sformat( argument.name, sformat.green ) + ": " + custom_printer_impl( argument.value, "\n  " )
    print( s )


def infox( *arg, frame_depth = 2, **kwargs ):
    f = debug._process( arg, kwargs, frame_depth = frame_depth )
    s = sformat( f"{ f.filename }:{ f.lineno }:", sformat.blue )
    for argument in f.arguments:
        if s:
            s += "\n  "
        else:
            s += "  "
        s += sformat( argument.name, sformat.green ) + ": " + custom_printer_impl( argument.value, "\n  " )
    print( s )
    sys.exit( 0 )
