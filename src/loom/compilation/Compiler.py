"""Les compilateurs : ce qui transforme un source généré en bibliothèque chargeable, PAR DEVICE.

Un device dit avec quoi il se compile (`Device.compiler`) : le compilateur hôte pour le CPU, `nvcc`
autour du compilateur hôte pour CUDA. C'est le seul aiguillage -- `make_library` ne sait pas quel
device il sert, il demande une commande au compilateur et gère le cache disque, pareil pour tous.

Un compilateur répond à trois questions :
  * `ninja_rules()` / `rule_for( src )` -- comment compiler une source en objet, lier une
    bibliothèque, un exécutable (voir `build.py`, qui ne connaît que des chemins et des règles) ;
  * `build_signature` -- ce qui, HORS du source, change le binaire (les flags, la machine quand
    `-march=native` en fait partie) ; `JaxFfi` la met dans le nom du `.so` avec le hash du source,
    pour qu'un changement de réglage ne retombe pas sur un cache bâti avec un autre ;
  * `describe()` -- ce que `sdot-toolchain` affiche.

Réglages d'environnement, communs :
  * `SDOT_CXX`     : le compilateur hôte (sinon `CXX`, sinon `c++` / `clang++` / `g++` sur PATH).
  * `SDOT_CXXFLAGS`: des flags de plus, découpés en mots -- l'échappatoire pour essayer un réglage
                     sans toucher au code. Ils entrent dans la signature.
  * `SDOT_NO_MARCH_NATIVE=1` : revenir au x86-64 de base (un `.so` à emporter ailleurs).
  * `LOOM_LINEINFO=1`     : `-g`, pour qu'un profileur / sanitizer nomme la ligne.
  * `LOOM_BOUNDS_CHECK=1` : arme la vérification de bornes de `TensorView::squeeze` (voir
                            `common_macros.h`). Un test par accès, réservé au diagnostic.
"""
from pathlib import Path
import platform
import shutil
import shlex
import sys
import os


def env_cxxflags() -> list:
    """`SDOT_CXXFLAGS`, découpé en mots."""
    return shlex.split( os.getenv( "SDOT_CXXFLAGS", "" ) )


def cpu_model() -> str:
    """Le processeur, tel que `-march=native` le voit -- ce qui doit entrer dans le nom d'un `.so`
    compilé pour lui. `/proc/cpuinfo` sur Linux, `platform` ailleurs."""
    try:
        with open( "/proc/cpuinfo" ) as f:
            for line in f:
                if line.lower().startswith( "model name" ) or line.lower().startswith( "flags" ):
                    return line.split( ":", 1 )[ 1 ].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def find_host_cxx() -> str | None:
    """Le compilateur C++ hôte : `SDOT_CXX`, puis `CXX`, puis les noms usuels sur PATH."""
    for var in ( "SDOT_CXX", "CXX" ):
        cxx = os.getenv( var )
        if cxx and ( shutil.which( cxx ) or Path( cxx ).is_file() ):
            return cxx
    for name in ( "c++", "clang++", "g++" ):
        if shutil.which( name ):
            return name
    return None


class Compiler:
    """Le contrat. Une instance par device, obtenue par `Device.compiler`."""

    name = "compiler"

    def is_available( self ) -> bool:
        raise NotImplementedError

    @property
    def build_signature( self ) -> str:
        raise NotImplementedError

    def ninja_rules( self ) -> dict:
        """nom -> ( commande ninja, a un depfile, description ). Attendus : une règle par sorte de source
        (`rule_for`), `link_shared`, `link_executable`. Variables disponibles dans une commande
        de compilation : `$includes`, `$defines`, `$extra` ; de liaison : `$libs`, `$soname`."""
        raise NotImplementedError

    def rule_for( self, src: Path ) -> str:
        """La règle qui compile cette source (par extension : `.cpp` -> le compilateur hôte,
        `.cu` -> nvcc)."""
        raise NotImplementedError

    def link_libraries( self, libraries: list ) -> str:
        """Les flags pour lier ces bibliothèques partagées (chemins), rpath compris."""
        raise NotImplementedError

    def soname_flags( self, out: Path ) -> str:
        return ""

    def source_suffix( self ) -> str:
        return ".cpp"

    def library_file_name( self, name: str ) -> str:
        return f"lib{ name }.dylib" if sys.platform == "darwin" else f"lib{ name }.so"

    def describe( self ) -> list:
        """Lignes (`nom`, `valeur`) pour `sdot-toolchain`."""
        raise NotImplementedError


class HostCxx( Compiler ):
    """Le compilateur hôte, tel quel : ce que le CPU utilise.

    `-O3 -march=native` par défaut. Ce n'était PAS le cas tant que le clip était scalaire : mesuré,
    `-march=native` ne rendait rien (voir `notes/2026-09-02-perf-2d-lmo.md`), et il figeait
    l'architecture de la machine dans un `.so` que le cache nommait d'après le seul `.cpp` généré.
    Le noyau à registres de `sdot/cell/Moteur2Reg.h` change la donne : écrit en asimd, il tient
    huit sommets dans un registre AVX2 et se DÉCOUPE en deux `xmm` SSE2 sans ce flag -- mesuré,
    1e6 germes 2D sur le même Xeon : 1.00 s en x86-64 de base, 0.33 s avec `-march=native`. Le
    piège du cache est levé autrement : le nom du `.so` porte les flags ET le modèle de processeur
    (`build_signature`).

    `-fvisibility=hidden` : une bibliothèque générée n'exporte que son point d'entrée (déclaré
    `visibility( "default" )` par le source généré) -- mesuré, le `.so` d'un noyau passe de 5.5 Mo
    à 0.3 Mo, et rien de ses milliers d'instanciations de templates n'est visible d'une autre.

    `SDOT_CXXFLAGS` qui nomme déjà un `-march=` / `-mcpu=` l'emporte.
    """

    name = "host c++"

    def __init__( self, cxx: str | None = None ):
        self.cxx = cxx or find_host_cxx()

    def is_available( self ) -> bool:
        return self.cxx is not None

    def march_flags( self ) -> list:
        if os.environ.get( "SDOT_NO_MARCH_NATIVE" ):
            return []
        if any( f.startswith( "-march" ) or f.startswith( "-mcpu" ) for f in env_cxxflags() ):
            return []
        return [ "-march=native" ]

    def opt_flags( self ) -> list:
        # `-O3` vaut 4 % sur le corps du kernel (Xeon W-2145, 16 threads, FP64, 1e6 germes en 2D,
        # leaf = 10 : 1.018 s en `-O2`, 0.979 s en `-O3`).
        return [ "-O3", "-fno-math-errno" ]

    def diagnostic_flags( self ) -> list:
        flags = []
        if os.environ.get( "LOOM_LINEINFO" ):
            flags.append( "-g" )
        if os.environ.get( "LOOM_BOUNDS_CHECK" ):
            flags.append( "-DLOOM_BOUNDS_CHECK" )
        return flags

    def flags( self ) -> list:
        return [ "-std=c++20", *self.opt_flags(), *self.march_flags(), *self.diagnostic_flags(),
                 "-fPIC", "-pthread", "-fvisibility=hidden", "-fvisibility-inlines-hidden", *env_cxxflags() ]

    @property
    def build_signature( self ) -> str:
        flags = self.flags()
        sig = f"{ self.cxx }|" + " ".join( flags )
        if "-march=native" in flags:
            sig += "|" + cpu_model()
        return sig

    def _require( self ):
        if self.cxx is None:
            raise RuntimeError( "sdot: aucun compilateur C++ trouvé (SDOT_CXX, CXX, ou c++/clang++/g++ sur PATH)" )

    def ninja_rules( self ):
        self._require()
        flags = " ".join( self.flags() )
        # ELF : lier chaque référence INTERNE à la définition locale -- pas de PLT pour les appels
        # d'une bibliothèque générée à ses propres instanciations de templates.
        bsymbolic = "" if sys.platform == "darwin" else "-Wl,-Bsymbolic"
        return {
            "cxx":             ( f"{ self.cxx } { flags } $includes $defines $extra -MMD -MF $out.d -c $in -o $out", True, "c++ $in $defines" ),
            "link_shared":     ( f"{ self.cxx } -pthread -shared { bsymbolic } $soname -o $out $in $libs", False, "link $out" ),
            "link_executable": ( f"{ self.cxx } -pthread -o $out $in $libs", False, "link $out" ),
        }

    def rule_for( self, src ):
        return "cxx"

    def source_suffix( self ) -> str:
        """L'extension d'un source généré pour ce compilateur (`.cu` sous nvcc)."""
        return ".cpp"

    def link_libraries( self, libraries ):
        flags = []
        for lib in libraries:
            lib = Path( lib )
            name = lib.name
            for prefix, suffix in ( ( "lib", ".so" ), ( "lib", ".dylib" ) ):
                if name.startswith( prefix ) and name.endswith( suffix ):
                    name = name[ len( prefix ):-len( suffix ) ]
            flags += [ f"-L{ lib.parent }", f"-l{ name }", f"-Wl,-rpath,{ lib.parent }" ]
        return " ".join( flags )

    def soname_flags( self, out ):
        return f"-Wl,-install_name,@rpath/{ Path( out ).name }" if sys.platform == "darwin" else ""

    def describe( self ):
        return [ ( "c++ hôte", self.cxx or "introuvable" ), ( "flags", " ".join( self.flags() ) ) ]


def find_nvcc() -> str | None:
    """`nvcc` : `SDOT_NVCC`, puis celui du paquet pip `nvidia-cuda-nvcc` (le même toolkit que le
    plugin CUDA de Jax, et un compilateur sans rien installer sur la machine), puis
    `/usr/local/cuda/bin`, puis PATH."""
    override = os.getenv( "SDOT_NVCC" )
    if override and Path( override ).is_file():
        return override
    import site
    for root in [ *site.getsitepackages(), site.getusersitepackages() ]:
        for p in sorted( Path( root ).glob( "nvidia/cu*/bin/nvcc" ), reverse = True ):
            if p.is_file():
                return str( p )
    for p in ( Path( "/usr/local/cuda/bin/nvcc" ), ):
        if p.is_file():
            return str( p )
    return shutil.which( "nvcc" )


class Nvcc( Compiler ):
    """`nvcc` autour du compilateur hôte : ce que CUDA utilise. Une source `.cu` passe par nvcc
    (le noyau, code hôte et device dans la même unité), une `.cpp` par le compilateur hôte tel
    quel ; nvcc lie (il sait où est `libcudart`).

    `arch` est celle de la carte présente (`sm_75`), lue par `CudaGpu.sm_arch` : on compile pour
    ELLE. Un binaire pour plusieurs architectures est l'affaire des wheels (étape 4).

    `--expt-relaxed-constexpr` : les fonctions `constexpr` de la bibliothèque standard
    (`std::min`, `std::forward`, `std::tuple`, ...) deviennent appelables du device -- ce que le
    code des noyaux fait partout. `--extended-lambda` : un corps écrit à la main peut rester un
    lambda, à condition de le marquer `[] HD ( ... )` (ce que loom génère, lui, est un foncteur).
    """

    name = "nvcc"

    def __init__( self, host: HostCxx | None = None, arch: str | None = None ):
        self.host = host or HostCxx()
        self.nvcc = find_nvcc()
        self.arch = arch or os.getenv( "SDOT_CUDA_ARCH" ) or "native"

    def is_available( self ) -> bool:
        return self.nvcc is not None and self.host.is_available()

    def source_suffix( self ) -> str:
        return ".cu"

    def flags( self ) -> list:
        # les `-D` valent pour les deux passes (nvcc les prend directement) ; le reste des flags
        # hôte va au compilateur hôte via `-Xcompiler`
        host_flags = [ f for f in self.host.flags() if f not in ( "-std=c++20", "-pthread" ) and not f.startswith( "-D" ) ]
        defines    = [ f for f in self.host.flags() if f.startswith( "-D" ) ]
        # `--Werror cross-execution-space-call` : appeler une fonction hôte depuis du code device est
        # une ERREUR de compilation, pas un avertissement -- nvcc en fait sinon un piège qui se
        # manifeste à l'exécution en « illegal memory access », loin de la ligne fautive
        return [ "-std=c++20", f"-ccbin={ self.host.cxx }", f"-arch={ self.arch }", "-O3", *defines,
                 "--expt-relaxed-constexpr", "--extended-lambda", "--Werror", "cross-execution-space-call",
                 "-Xcompiler", ",".join( host_flags ),
                 *( [ "-lineinfo" ] if os.environ.get( "LOOM_LINEINFO" ) else [] ) ]

    @property
    def build_signature( self ) -> str:
        return f"nvcc:{ self.nvcc }|" + " ".join( self.flags() ) + "|" + self.host.build_signature

    def ninja_rules( self ):
        if not self.is_available():
            raise RuntimeError( "sdot: nvcc introuvable (pip install nvidia-cuda-nvcc-cu13, ou SDOT_NVCC=/chemin/nvcc)" )
        rules = dict( self.host.ninja_rules() )
        flags = " ".join( self.flags() )
        bsymbolic = "" if sys.platform == "darwin" else "-Xlinker -Bsymbolic"
        rules[ "nvcc" ]            = ( f"{ self.nvcc } { flags } $includes $defines $extra -MD -MF $out.d -c $in -o $out", True, "nvcc $in $defines" )
        rules[ "link_shared" ]     = ( f"{ self.nvcc } -ccbin={ self.host.cxx } -shared { bsymbolic } $soname -o $out $in $libs", False, "link $out" )
        rules[ "link_executable" ] = ( f"{ self.nvcc } -ccbin={ self.host.cxx } -o $out $in $libs", False, "link $out" )
        return rules

    def rule_for( self, src ):
        return "nvcc" if Path( src ).suffix == ".cu" else "cxx"

    def link_libraries( self, libraries ):
        # nvcc ne connaît pas `-Wl,` : ce qui va à l'éditeur de liens passe par `-Xlinker`
        return self.host.link_libraries( libraries ).replace( "-Wl,-rpath,", "-Xlinker -rpath -Xlinker " )

    def soname_flags( self, out ):
        return self.host.soname_flags( out ).replace( "-Wl,-install_name,", "-Xlinker -install_name -Xlinker " )

    def describe( self ):
        return [ ( "nvcc", self.nvcc or "introuvable" ), ( "arch", self.arch ), *self.host.describe() ]
