"""The build graph: units (`.cpp` + defines -> `.o`), libraries and the executables that link
them, and ninja to redo only what changed.

Why ninja and not a home-made hash: a kernel is not a file but a CLOSURE of headers, and only the
compiler knows it exactly (`-MMD`). ninja reads those depfiles, compares timestamps, and rebuilds a
unit if and only if one of ITS headers moved -- whereas the global hash of the whole tree
(`cpp_sources_hash`, formerly) rebuilt every kernel because of a comment touched in a header it did
not include. Measured: 88 ms of graph per kernel against ~8 s of compilation, i.e. 1 % -- a good
deal for the one hard thing.

= TWO GRAPHS, because there are two kinds of targets

The criterion is not "common / specific" but HAVING DEPENDENTS OR NOT.

  * The COMMON graph (`build/`) holds what has some: the runtime library (the thread queue, one
    per compiler signature) and the DOMAIN UNITS -- a source compiled once per
    (source, defines, compiler) and linked by all the kernels that name it. It is permanent, it
    has a lock, and it no longer grows with the number of kernels.
  * A kernel's OWN graph (`build/kernels/<target>/`) holds its generated source, its object and
    its library: one kernel, zero dependents, and a name that is already a hash of its content.
    Two edges, its own `build.ninja`, its own lock, its own dependency log.

What this changes, and this is the point: FORGETTING A KERNEL BECOMES A DELETION. `rm -rf` of its
directory, with no graph mutation, no lock, no possible orphan edge -- whereas a single
manifest could only grow (5104 edges, 3506 libraries, 2.7 GB measured before this
partition). And the lock decomposes: it now serializes only the "is the runtime up to date?" phase,
so that two processes compiling two DIFFERENT kernels no longer wait for each other.
Two processes on the SAME kernel still wait for each other, which is the only case where they must.

The two phases are sequential and in that order: the common one first (it may refresh the
runtime), the kernel next, which then sees a newer input and relinks. The common artifacts
enter the kernel's graph as plain INPUT FILES, with no rule -- this is what
prevents two kernel directories from independently deciding to rebuild the runtime and fighting
over the same output file. (It is also why this is not a `subninja`.)

The commands come from the device's compiler (`Compiler.commands`): it is the one that knows how to
compile a `.cpp` or a `.cu`, and it hands them out as ARGV TEMPLATES with named holes (`{in}`,
`{out}`, `{depfile}`, ...). This module turns them into ninja SYNTAX (`_to_ninja_syntax`); the same
template could be run directly or written into a `compile_commands.json`. The flags belong to the
compiler, the syntax to whoever renders the command -- and neither layer knows the other.
"""
from pathlib import Path
import subprocess
import hashlib
import shutil
import json
import sys
import os

from . import build_dir, include_dirs, _dev_repo_root
from ..util.encode_base_62 import encode_base_62
from .. import env


def ninja_path() -> str:
    p = env.var( "NINJA" ) or shutil.which( "ninja" )
    if p is None:
        # the `ninja` pip package ships the binary next to the interpreter's scripts
        candidate = Path( sys.executable ).parent / "ninja"
        if candidate.is_file():
            p = str( candidate )
    if p is None:
        raise RuntimeError( "loom : `ninja` not found (pip install ninja, or LOOM_NINJA=/path/ninja)" )
    return p


def kernels_root() -> Path:
    """Where a per-kernel directory lives -- each is erased in one go (see the module docstring)."""
    return build_dir() / "kernels"


def _short_hash( *parts ) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update( str( p ).encode() )
        h.update( b"\0" )
    return encode_base_62( h.hexdigest() )[ :10 ]


# the holes of an argv template (`Compiler.commands`), rendered as ninja variables. `{depfile}` is
# `$out.d`: ninja wants the depfile next to the output, and declares it in the rule.
_NINJA_PLACEHOLDERS = {
    "{in}": "$in", "{out}": "$out", "{depfile}": "$out.d",
    "{includes}": "$includes", "{defines}": "$defines", "{extra}": "$extra",
    "{libs}": "$libs", "{soname}": "$soname",
}


def _to_ninja_syntax( argv ) -> str:
    """An argv template as a ninja command line. Arguments that are not holes
    pass through unchanged."""
    return " ".join( _NINJA_PLACEHOLDERS.get( a, a ) for a in argv )


def _ninja_escape( s: str ) -> str:
    return str( s ).replace( "$", "$$" ).replace( " ", "$ " ).replace( ":", "$:" )


class Manifest:
    """The graph on disk. `rules`: name -> { command, depfile?, deps? }; `edges`: output ->
    { rule, inputs, implicit, vars }. Loaded and rewritten under the lock."""

    def __init__( self, root: Path ):
        self.root = Path( root )
        self.path = self.root / "ninja" / "manifest.json"
        self.rules = {}
        self.edges = {}
        if self.path.is_file():
            data = json.loads( self.path.read_text() )
            self.rules = data.get( "rules", {} )
            self.edges = data.get( "edges", {} )

    def add_rule( self, name: str, command: str, depfile: bool, description: str ):
        rule = { "command": command, "description": description }
        if depfile:
            rule[ "depfile" ] = "$out.d"
            rule[ "deps" ] = "gcc"
        self.rules[ name ] = rule

    def add_edge( self, out: Path, rule: str, inputs: list, implicit: list = (), **variables ):
        self.edges[ str( out ) ] = { "rule": rule, "inputs": [ str( i ) for i in inputs ],
                                     "implicit": [ str( i ) for i in implicit ],
                                     "vars": { k: str( v ) for k, v in variables.items() if v } }

    def prune( self, keep = () ) -> int:
        """Removes the edges whose OUTPUT no longer exists, except `keep` (what this build has just
        declared and is about to build).

        Always safe: if someone still needs a removed edge, the process that declares it
        will put it back. This is what makes deleting a kernel directory -- or an artifact by
        hand -- CLEAN the graph instead of letting it swell, and what was missing when everything
        lived in a single manifest."""
        keep = { str( g ) for g in keep }
        live = { out: e for out, e in self.edges.items()
                     if out in keep or Path( out ).exists() }
        removed = len( self.edges ) - len( live )
        self.edges = live
        return removed

    def save( self ):
        self.path.parent.mkdir( parents = True, exist_ok = True )
        tmp = self.path.with_suffix( ".json.tmp" )
        tmp.write_text( json.dumps( { "rules": self.rules, "edges": self.edges }, indent = 1 ) )
        tmp.replace( self.path )

    def write_ninja( self ) -> Path:
        lines = [ "# generated by loom.compilation.build -- do not edit, see ninja/manifest.json", "ninja_required_version = 1.3", "" ]
        for name, rule in self.rules.items():
            lines.append( f"rule { name }" )
            lines.append( f"  command = { rule[ 'command' ] }" )
            lines.append( f"  description = { rule.get( 'description', '$out' ) }" )
            if "depfile" in rule:
                lines.append( f"  depfile = { rule[ 'depfile' ] }" )
                lines.append( f"  deps = { rule[ 'deps' ] }" )
            lines.append( "" )
        for out, edge in self.edges.items():
            ins = " ".join( _ninja_escape( i ) for i in edge[ "inputs" ] )
            imp = " ".join( _ninja_escape( i ) for i in edge[ "implicit" ] )
            lines.append( f"build { _ninja_escape( out ) }: { edge[ 'rule' ] } { ins }" + ( f" | { imp }" if imp else "" ) )
            for k, v in edge[ "vars" ].items():
                lines.append( f"  { k } = { v }" )
        path = self.root / "build.ninja"
        content = "\n".join( lines ) + "\n"
        if not ( path.exists() and path.read_text() == content ):
            path.write_text( content )
        return path


class _Lock:
    """A file lock around a graph and its ninja (POSIX; no-op elsewhere).

    One per graph: the common one serializes "is the runtime up to date?", a kernel's one
    serializes the compilation of THAT kernel. Two different kernels no longer cross paths."""

    def __init__( self, root: Path ):
        self.path = Path( root ) / "ninja" / "lock"
        self.fd = None

    def __enter__( self ):
        self.path.parent.mkdir( parents = True, exist_ok = True )
        self.fd = os.open( self.path, os.O_RDWR | os.O_CREAT, 0o644 )
        try:
            import fcntl
            fcntl.flock( self.fd, fcntl.LOCK_EX )
        except ImportError:
            pass
        return self

    def __exit__( self, *exc ):
        try:
            import fcntl
            fcntl.flock( self.fd, fcntl.LOCK_UN )
        except ImportError:
            pass
        os.close( self.fd )


def force_build() -> bool:
    """`LOOM_FORCE_BUILD`: rebuild the requested targets even if ninja believes them up to date -- for
    when the toolchain changed in a way the depfiles do not see (a flag, a tool)."""
    return env.flag( "FORCE_BUILD" )


class _Graph:
    """A graph in a directory: what THIS build declares in it, plus what is needed to merge it
    into the manifest already there and run ninja on it.

    The declarations stay here and are not written right away: the manifest on disk
    is only read and rewritten in `run`, under the lock -- so the edges another process
    may have added in the meantime are not lost (formerly, a lock held for the whole duration of
    the build made the question moot, at the price of total serialization)."""

    def __init__( self, root, sig: str, commands: dict ):
        self.root = Path( root )
        self.sig = sig
        self.rules = { f"{ name }_{ sig }": ( _to_ninja_syntax( argv ), depfile, description )
                       for name, ( argv, depfile, description ) in commands.items() }
        self.edges = {}

    def add_edge( self, out: Path, rule: str, inputs: list, implicit: list = (), **variables ):
        self.edges[ str( out ) ] = dict( rule = rule, inputs = list( inputs ),
                                         implicit = list( implicit ), variables = variables )

    def run( self, targets: list ):
        """Merges our declarations into the directory's manifest, then builds `targets`."""
        targets = [ str( t ) for t in targets ]
        if not targets:
            return
        with _Lock( self.root ):
            manifest = Manifest( self.root )
            for name, ( command, depfile, description ) in self.rules.items():
                manifest.add_rule( name, command, depfile, description )
            for out, e in self.edges.items():
                manifest.add_edge( out, e[ "rule" ], e[ "inputs" ], e[ "implicit" ], **e[ "variables" ] )
            manifest.prune( keep = self.edges )
            manifest.save()
            ninja_file = manifest.write_ninja()

            if force_build():
                for t in targets:
                    Path( t ).unlink( missing_ok = True )
                    for i in manifest.edges.get( t, {} ).get( "inputs", [] ):
                        if i.endswith( ".o" ):
                            Path( i ).unlink( missing_ok = True )

            # `LOOM_BUILD_JOBS`: the parallelism (default: ninja's, all cores) -- a CUDA
            # catalog compiles a hundred units of a gigabyte each, we do not want them all
            # at the same time on a shared machine
            jobs = env.var( "BUILD_JOBS" )
            cmd = [ ninja_path(), "-C", str( self.root ), "-f", str( ninja_file ),
                    *( [ "-j", jobs ] if jobs else [] ), *targets ]
            r = subprocess.run( cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True )
            out = r.stdout or ""
            # ninja's own line for a no-op build is noise; a real compilation is worth seeing
            if "no work to do" not in out:
                print( out, end = "", flush = True )
            if r.returncode:
                raise RuntimeError( f"ninja failed ({ r.returncode }) on { targets }" )


class Build:
    """A build session: units, libraries and executables are declared in it,
    then `run( targets )` does what is necessary.

    `work_dir` is the OWN directory of what is being built -- a kernel's, erasable in one go.
    Without it, everything goes into the common graph: this is what the catalog does (a single big link in
    a throwaway directory) and the C++ tests."""

    def __init__( self, device, work_dir = None ):
        self.device = device
        self.compiler = device.compiler
        self.root = build_dir()
        self.sig = _short_hash( self.compiler.build_signature )
        commands = self.compiler.commands()
        self.common = _Graph( self.root, self.sig, commands )
        self.own = _Graph( work_dir, self.sig, commands ) if work_dir is not None else self.common

    def close( self ):
        pass

    def __enter__( self ):
        return self

    def __exit__( self, *exc ):
        self.close()

    # ── declarations ─────────────────────────────────────────────────────────
    def _rule( self, name ) -> str:
        return f"{ name }_{ self.sig }"

    def object( self, src: Path, defines: dict = None, extra_flags: list = (), shared = True,
                include_first: list = () ) -> Path:
        """The object of a source compiled with these `defines` -- one unit per (source, defines,
        compiler).

        `include_first`: `-I` roots searched BEFORE the common ones -- a kernel's own overlay of
        generated headers (see `generated_headers.py`), which must win over the shared tree.

        `shared` (the default): the unit has dependents, the same `.o` serves all the kernels that
        ask for it, so it lives in the COMMON graph. `shared = False`: the object belongs to nobody
        else (a kernel's generated source), it lives in the own directory and goes away with it."""
        src = Path( src ).resolve()
        defines = dict( defines or {} )
        extra_flags = list( extra_flags )
        include_first = [ str( d ) for d in include_first ]
        graph = self.common if shared else self.own
        name = f"{ src.stem }_{ _short_hash( self.sig, src, sorted( defines.items() ), extra_flags, *include_first ) }.o"
        obj = ( graph.root / "obj" / name ) if graph is self.common else ( graph.root / name )
        graph.add_edge(
            obj, self._rule( self.compiler.rule_for( src ) ), [ src ],
            includes = " ".join( f"-I { _ninja_escape( d ) }" for d in [ *include_first, *include_dirs() ] ),
            defines  = " ".join( f"-D{ k }={ v }" if v is not None else f"-D{ k }" for k, v in defines.items() ),
            extra    = " ".join( extra_flags ),
        )
        return obj

    def shared_library( self, out: Path, objects: list, libraries: list = (), shared = False ) -> Path:
        """`out` = a `.so` linked from `objects` and `libraries` (`.so` files of the graph: the
        directory of each goes into the rpath).

        `shared = True` for a library that has dependents (the runtime): it goes into the
        common graph."""
        out = Path( out )
        graph = self.common if shared else self.own
        graph.add_edge( out, self._rule( "link_shared" ), objects, implicit = libraries,
                         libs = self.compiler.link_libraries( libraries ),
                         soname = self.compiler.soname_flags( out ) )
        return out

    def executable( self, out: Path, objects: list, libraries: list = () ) -> Path:
        out = Path( out )
        self.own.add_edge( out, self._rule( "link_executable" ), objects, implicit = libraries,
                              libs = self.compiler.link_libraries( libraries ) )
        return out

    def runtime_library( self ) -> Path:
        """`libloom_runtime` for this compiler: the process's thread queue, and everything
        the kernels share without having to recompile it. One per compiler signature, in the
        common graph -- it is the very example of a target with dependents."""
        objects = [ self.object( src ) for src in runtime_sources() ]
        return self.shared_library(
            self.common.root / "runtime" / self.compiler.library_file_name( f"loom_runtime_{ self.sig }" ),
            objects, shared = True )

    # ── run ──────────────────────────────────────────────────────────────────
    def run( self, targets: list ):
        """Builds `targets` (and what they depend on), and nothing else.

        Two phases, in this order: the COMMON one first -- it may refresh the runtime or a
        domain unit -- then the OWN one, which then sees a newer input and relinks. The
        common artifacts enter the own graph as plain input files, with no
        rule: no kernel directory can decide to rebuild the runtime."""
        if self.own is self.common:
            self.common.run( targets )
            return
        self.common.run( list( self.common.edges ) )
        self.own.run( targets )


def runtime_sources() -> list:
    """The sources of `libloom_runtime` (`loom/cpp/runtime/*.cpp`), from a checkout or a wheel."""
    dev_root = _dev_repo_root()
    if dev_root is not None:
        root = dev_root / "loom" / "cpp" / "runtime"
    else:
        root = Path( __file__ ).resolve().parents[ 1 ] / "_cpp" / "runtime"
    return sorted( root.glob( "*.cpp" ) )
