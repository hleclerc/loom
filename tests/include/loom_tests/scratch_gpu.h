#pragma once

#include <loom/support/atomic_add.h>
#include <loom/support/common_macros.h>
#include <type_traits>

/// Les deux foncteurs du spike `scratch` côté GPU. Ils sont ici et pas dans le corps Python parce
/// qu'un noyau lancé a besoin d'un foncteur au niveau du NAMESPACE ( C++ interdit les méthodes
/// template dans une classe locale ), et que le corps d'un `FfiCode.handler` est une suite
/// d'instructions dans le handler.
namespace loom_tests {

/// recopie dans le scratch ce qui est positif ( zéro sinon )
struct GarderPositifs {
    template<class T_tmp,class T_src>
    HD void operator()( auto id, T_tmp tmp, T_src src ) const {
        // NB le type est EXPLICITE : `auto v = src( id ); v > 0 ? v : 0` tronque en silence --
        // le ternaire cherche un type commun entre l'accesseur et le littéral `0`, et le trouve
        // entier. Piège facile, et muet : le résultat reste plausible.
        double v = src( id );
        tmp( id ) = v > 0 ? v : 0.0;
    }
};

/// somme le scratch dans la sortie. NB `atomic_add` de loom ne couvre pas `SI` sur CUDA
/// ( `atomicAdd` n'a pas de surcharge 64 bits signée ), d'où une somme en `double`.
struct Sommer {
    template<class T_out,class T_tmp>
    HD void operator()( auto id, T_out out, T_tmp tmp ) const {
        auto &o = out( 0 ).ref();
        sdot::atomic_add( o, std::remove_reference_t<decltype( o )>( tmp( id ) ) );
    }
};

/// compte, sur la carte, combien d'entrées sont positives
struct Compter {
    template<class T_cpt,class T_src>
    HD void operator()( auto id, T_cpt cpt, T_src src ) const {
        double v = src( id );
        if ( v > 0 ) {
            auto &c = cpt( 0 ).ref();
            sdot::atomic_add( c, std::remove_reference_t<decltype( c )>( 1 ) );
        }
    }
};

/// compacte les positifs dans `dst`, en réservant sa place par un ticket atomique
struct Compacter {
    template<class T_dst,class T_cpt,class T_src>
    HD void operator()( auto id, T_dst dst, T_cpt cpt, T_src src ) const {
        double v = src( id );
        if ( v > 0 ) {
            auto &c = cpt( 0 ).ref();
            auto k = sdot::atomic_fetch_add( c, std::remove_reference_t<decltype( c )>( 1 ) );
            if ( k < dst.shape( 0 ) )
                dst( k ) = v;
        }
    }
};

/// pose un SCALAIRE venu de l'hôte dans une sortie. Passer un scalaire nu à `run_parallel` est
/// exactement ce qui faisait tomber le frontal de nvcc avant `KernelFormProbe`.
struct Poser {
    template<class T_out,class T_v>
    HD void operator()( auto, T_out out, T_v v ) const { out( 1 ) = v; }
};

} // namespace loom_tests
