#pragma once

#include "../containers/TensorView.h"
#include "../common_types.h"
#include <cstdlib>
#include <vector>

namespace sdot {

/// De la mémoire allouée PENDANT l'appel, à une taille que seul le noyau connaît.
///
/// On avait supposé la chose impossible : Jax préalloue le gros de la RAM du GPU, donc il ne
/// resterait rien. La prémisse est fausse -- et surtout elle porte sur la mauvaise contrainte.
/// Ce qu'XLA exige au traçage, c'est la forme d'une SORTIE. Une taille INTERNE, il ne l'a jamais
/// demandée. Donc tout ce qui est construit et consommé dans le même appel ( un index, un tampon
/// de tri, un CSR ) peut être dimensionné EXACTEMENT, sans borne devinée, et sous `jit` comme en
/// eager. C'est la différence avec `scratch_attributes`, dont la capacité descend de Python.
///
/// D'OÙ VIENT LA MÉMOIRE, et pourquoi ce n'est pas la même réponse partout :
///
///   * sur GPU, il faut le pool d'XLA -- c'est lui qui détient la carte. `XLA_FFI_DeviceMemory_
///     Allocate` est un champ de la struct `XLA_FFI_Api`, donc de l'ABI C stable.
///   * sur CPU, non : le handler tourne sur l'hôte, ses tampons SONT de la mémoire hôte, et un
///     `malloc` fait l'affaire. Ce n'est pas un pis-aller -- le backend CPU d'XLA répond
///     « No device memory allocator available on this platform », et il a raison : il n'y a pas
///     de device.
///
/// Le choix est celui du DEVICE ( `Device.cpp_scratch_decl` ), pas du corps : un corps écrit une
/// fois marche des deux côtés.
///
/// L'ÉCHEC EST UNE VALEUR, pas une exception : un pool peut dire non. `view()` rend alors une vue
/// VIDE et lève `failed`, que le handler engendré rapporte à XLA en sortant. Un corps qui borne
/// ses boucles sur LA VUE -- et non sur ce qu'il voulait y mettre -- ne fait alors rien, ce qui
/// est le comportement voulu. Un corps qui borne sur son intention écrit à travers un pointeur
/// nul : c'est exactement le segfault qui a servi à découvrir le refus du CPU, donc l'erreur est
/// facile à faire et vaut d'être dite ici.
template<class _MemorySpace>
struct Scratch {
    using        MemorySpace = _MemorySpace;

    /// comment on alloue, rendu opaque EXPRÈS : ce header ne connaît pas XLA ( il est inclus par
    /// des en-têtes écrits à la main, qui n'ont pas à traîner `xla/ffi/api/ffi.h` ). C'est la
    /// source engendrée qui branche ces deux pointeurs.
    using        AllocFn     = void *( * )( void *ctx, SI nb_bytes, SI alignment );
    /// `nullptr` quand le pool libère tout seul à la fin de l'appel ( cas d'XLA ).
    using        FreeFn      = void ( * )( void *ctx, void *ptr );

    /* */        Scratch     ( AllocFn alloc_fn, void *ctx = nullptr, FreeFn free_fn = nullptr )
                                : alloc_fn( alloc_fn ), free_fn( free_fn ), ctx( ctx ) {}
    /* */        Scratch     ( const Scratch & ) = delete;
    Scratch&     operator=   ( const Scratch & ) = delete;

    /* */        ~Scratch    () {
        if ( free_fn )
            for ( void *p : blocs )
                free_fn( ctx, p );
    }

    /// `n` éléments de type `T`, contigus, NON initialisés -- comme un `malloc`, et pour la même
    /// raison : semer coûte, et l'appelant sait s'il écrit avant de lire.
    template<class T>
    auto         view        ( SI n ) {
        T *ptr = nullptr;
        if ( n > 0 ) { // 0 ne s'alloue pas : rien à demander, donc rien à refuser
            ptr = reinterpret_cast<T *>( alloc_fn( ctx, n * SI( sizeof( T ) ), SI( alignof( T ) ) ) );
            if ( ptr == nullptr ) {
                failed = true;
                n = 0;
            } else if ( free_fn )
                blocs.push_back( ptr );
        }
        return tensor_view<MemorySpace>( ptr, tuple( n ) );
    }

    AllocFn             alloc_fn;
    FreeFn              free_fn;
    void               *ctx;
    std::vector<void *> blocs;          ///< vide quand le pool libère lui-même
    bool                failed = false; ///< un refus au moins ( lu par le handler engendré )
};

/// le scratch d'un handler HÔTE : `aligned_alloc`, libéré en sortant. Voir la docstring ci-dessus
/// pour pourquoi ce n'est pas un pis-aller sur CPU.
template<class MemorySpace>
Scratch<MemorySpace> host_scratch() {
    return Scratch<MemorySpace>(
        []( void *, SI nb_bytes, SI alignment ) -> void * {
            // `aligned_alloc` veut une taille multiple de l'alignement
            SI a = alignment < SI( sizeof( void * ) ) ? SI( sizeof( void * ) ) : alignment;
            SI s = ( nb_bytes + a - 1 ) / a * a;
            return std::aligned_alloc( size_t( a ), size_t( s ) );
        },
        nullptr,
        []( void *, void *ptr ) { std::free( ptr ); }
    );
}

} // namespace sdot
