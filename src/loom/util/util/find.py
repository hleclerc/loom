
def find( iterable, predicate ):
    return next( ( x for x in iterable if predicate( x ) ), None )
