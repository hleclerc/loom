import hashlib
import struct

def encode_base_62( s: str, length: int = 11 ) -> str:
    chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    h = struct.unpack( '>Q', hashlib.sha256( s.encode() ).digest()[ :8 ] )[ 0 ]
    res = []
    for _ in range( length ):
        res.append( chars[ h % 62 ] )
        h //= 62
    return ''.join( reversed( res ) )
