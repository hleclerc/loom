#pragma once

#include "Tuple.h"  // IWYU pragma: export

namespace sdot {

/// Tuple with repetion of the last item
///   -> infinite size
template<class... Types>
class TupleRep {
public:
    /* */           TupleRep           ( Function, auto &&func, auto index ) : data( Function{}, FORWARD( func ), index ) {}
    /* */           TupleRep           ( Function, auto &&func ) : TupleRep( FORWARD( func ), 0_c ) {}

    /* */           TupleRep           ( Values, auto &&...values ) : data( Values{}, FORWARD( values )... ) {}

    /* */           TupleRep           ( const TupleRep &that ) = default;
    /* */           TupleRep           () = default;

    auto            operator[]         ( auto &&index ) const { return data[ min( index, Ct<int,sizeof...( Types )-1>() ) ]; }

    Tuple<Types...> data;
};

T_VT constexpr auto tuple_rep( T &&...a ) {
    return TupleRep<DECAYED_TYPE_OF( a )...>( Values(), a... );
}

} // namespace sdot

#include "Tuple.cxx" // IWYU pragma: export
