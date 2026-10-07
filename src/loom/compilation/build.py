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

= WHERE, AND HOW A NEW KERNEL APPEARS

The build directory is the user's cache of the platform (`build_dir()`), shared by every checkout and
every working directory: what a kernel is compiled against is in its NAME, not in where it was
launched from. A kernel that does not exist yet is built in a private directory
(`tmp/<pid>_<thread>/<kernel>`) and renamed into `kernels/` when complete (`KernelDir`): readers
only ever see finished kernels, and a crashed or concurrent build leaves nothing behind but a
temporary directory. For the rename to be harmless, a kernel's own graph is RELATIVE to its
directory (`_Graph.rel`): ninja runs from there, and nothing it writes names the place.

The commands come from the device's compiler (`Compiler.commands`): it is the one that knows how to
compile a `.cpp` or a `.cu`, and it hands them out as ARGV TEMPLATES with named holes (`{in}`,
`{out}`, `{depfile}`, ...). This module turns them into ninja SYNTAX (`_to_ninja_syntax`); the same
template could be run directly or written into a `compile_commands.json`. The flags belong to the
compiler, the syntax to whoever renders the command -- and neither layer knows the other.
"""
from contextlib import suppress
from pathlib import Path
import subprocess
import threading
import hashlib
import shutil
import time
import json
import sys
import os

from . import build_dir, include_dirs, loom_include_root, _dev_repo_root
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


def staging_root() -> Path:
    """Where a kernel is built before it exists: `<build>/tmp/<pid>_<thread>/<kernel>`."""
    return build_dir() / "tmp"


_STALE_STAGING_SECONDS = 24 * 3600
_staging_swept = False


def _sweep_staging():
    """Once per process: erase the staging directories a crashed process left behind (a day old and
    untouched -- a live build keeps writing into its own)."""
    global _staging_swept
    if _staging_swept:
        return
    _staging_swept = True
    cutoff = time.time() - _STALE_STAGING_SECONDS
    with suppress( OSError ):
        for d in staging_root().iterdir():
            with suppress( OSError ):
                if d.stat().st_mtime < cutoff:
                    shutil.rmtree( d, ignore_errors = True )


def locks_root() -> Path:
    """Where the lock of each kernel being built lives. Never swept: erasing a lock file that is
    held would let a second process take "the same" lock on a new file."""
    return build_dir() / "locks"


_LOCK_REPORT_SECONDS = 30


class _BuildLock:
    """An exclusive `flock` on `locks/<kernel>.lock`: one process builds a given new kernel, the
    others wait and then find it published.

    It is the OPERATING SYSTEM that holds the lock, on an open file: when its holder dies -- kill -9
    included -- the lock is gone, so a waiter is never stuck behind a crash and there is no liveness
    check to get wrong. The file itself stays, and says who holds it (pid, host, since when), for the
    message of those who wait. It saves duplicated work and nothing else: correctness is the
    business of the rename. Where `flock` does nothing (some network file systems, Windows) the
    worst case is the duplicated work."""

    def __init__( self, name: str ):
        self.path = locks_root() / f"{ name }.lock"
        self.fd = None

    def acquire( self, published ) -> bool:
        """Takes the lock, unless `published()` becomes true first (the holder finished). Returns
        whether it is held."""
        try:
            import fcntl
        except ImportError:
            return True
        self.path.parent.mkdir( parents = True, exist_ok = True )
        self.fd = os.open( self.path, os.O_RDWR | os.O_CREAT, 0o644 )
        started = last_report = time.monotonic()
        while True:
            try:
                fcntl.flock( self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB )
                break
            except OSError:
                pass
            if published():
                self.release()
                return False
            now = time.monotonic()
            if now - last_report >= _LOCK_REPORT_SECONDS:
                last_report = now
                print( f"loom: waiting for { self.holder() } to build { self.path.stem } "
                       f"({ now - started:.0f} s)", file = sys.stderr, flush = True )
            time.sleep( 0.05 )
        os.ftruncate( self.fd, 0 )
        os.pwrite( self.fd, f"pid { os.getpid() } on { os.uname().nodename if hasattr( os, 'uname' ) else '?' }\n".encode(), 0 )
        return True

    def holder( self ) -> str:
        try:
            return self.path.read_text().strip() or "another process"
        except OSError:
            return "another process"

    def release( self ):
        if self.fd is not None:
            os.close( self.fd )         # closing drops the flock
            self.fd = None


class KernelDir:
    """The directory of ONE kernel, from "being built" to "published".

        ws = KernelDir( name )
        with ws as work_dir:          # build in `work_dir`
            ...
        lib = ws.final / "<name>.so"  # and use it from `ws.final`

    A kernel that does not exist yet is built in a PRIVATE directory (`staging_root()/<pid>_<thread>/`)
    and RENAMED to `kernels_root()/<name>` once it is complete: `rename` is atomic, so nobody ever
    sees or loads a half-made kernel, whatever crashed or ran at the same time. Two processes that
    want the same new kernel each build their own copy -- the first rename wins, the other's copy
    is erased (same name = same content: it is the same kernel, and both can use the winner's).
    The directory stays valid across the move because its graph is relative to it (`_Graph.rel`).

    Before building, the process takes the kernel's lock (`_BuildLock`) and looks again: if another
    one was building it, it is published by now and this one only uses it. The existence of
    `kernels/<name>` is the witness that the build is complete -- it appears only by that rename --
    so no marker file is needed.

    A kernel that EXISTS is worked on in place, as before (ninja, the lock, the fast path): it is
    what happens when a header it includes has changed, and ninja is then the one to decide."""

    def __init__( self, name: str ):
        self.name = name
        self.final = kernels_root() / name
        self.staged = None
        self.lock = None

    def __enter__( self ) -> Path:
        if self.final.is_dir():
            return self.final
        self.lock = _BuildLock( self.name )
        if not self.lock.acquire( published = self.final.is_dir ):
            return self.final
        if self.final.is_dir():             # the one we waited for has published
            self.lock.release()
            return self.final
        _sweep_staging()
        self.staged = staging_root() / f"{ os.getpid() }_{ threading.get_ident() }" / self.name
        shutil.rmtree( self.staged, ignore_errors = True )      # a previous life of this pid
        self.staged.mkdir( parents = True )
        return self.staged

    def __exit__( self, exc_type, *_ ):
        if self.staged is None:
            return
        try:
            if exc_type is None:
                self.final.parent.mkdir( parents = True, exist_ok = True )
                try:
                    os.rename( self.staged, self.final )
                except OSError:
                    if not self.final.is_dir():     # not "someone published first": a real failure
                        raise
        finally:
            shutil.rmtree( self.staged, ignore_errors = True )
            with suppress( OSError ):
                self.staged.parent.rmdir()          # only if no other kernel of this thread is being built
            self.lock.release()


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
                     if out in keep or ( self.root / out ).exists() }
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

    def __init__( self, root, sig: str, commands: dict, relative = False ):
        self.root = Path( root )
        self.relative = relative
        self.sig = sig
        self.rules = { f"{ name }_{ sig }": ( _to_ninja_syntax( argv ), depfile, description )
                       for name, ( argv, depfile, description ) in commands.items() }
        self.edges = {}

    def rel( self, path ) -> str:
        """How this graph spells `path`. A RELATIVE graph (a kernel's own) spells what is under its
        root relative to it, and ninja runs from there (`-C`): nothing in its manifest, its logs
        or its commands then names the directory, so the directory can be built under a temporary
        name and renamed into place (`KernelDir`) without ninja noticing."""
        path = str( path )
        if self.relative:
            with suppress( ValueError ):
                return str( Path( path ).relative_to( self.root ) )
        return path

    def add_edge( self, out: Path, rule: str, inputs: list, implicit: list = (), **variables ):
        self.edges[ self.rel( out ) ] = dict( rule = rule, inputs = [ self.rel( i ) for i in inputs ],
                                              implicit = [ self.rel( i ) for i in implicit ], variables = variables )

    def run( self, targets: list ) -> bool:
        """Merges our declarations into the directory's manifest, then builds `targets`. Returns
        whether ninja had anything to do."""
        targets = [ self.rel( t ) for t in targets ]
        if not targets:
            return False
        with _Lock( self.root ):
            manifest = Manifest( self.root )
            for name, ( command, depfile, description ) in self.rules.items():
                manifest.add_rule( name, command, depfile, description )
            for out, e in self.edges.items():
                manifest.add_edge( out, e[ "rule" ], e[ "inputs" ], e[ "implicit" ], **e[ "variables" ] )
            manifest.prune( keep = self.edges )
            manifest.save()
            ninja_file = manifest.write_ninja()

            # THE WITNESS: ninja judges an output by its date, and a compiler or linker killed half way
            # (kill -9, power cut) leaves a truncated file NEWER than its inputs -- "up to date". So the
            # work is bracketed by a marker, under the lock: still there at the next run means the last
            # one never came back (a normal end, success or failure, removes it), and what it was making is
            # redone. The lock having been released says only that its holder is gone, not that it finished.
            marker = self.root / "ninja" / "building"
            crashed = marker.exists()
            if crashed:
                print( f"loom: a previous build in { self.root } did not finish, redoing { targets }",
                       file = sys.stderr, flush = True )
            if force_build() or crashed:
                for t in targets:
                    ( self.root / t ).unlink( missing_ok = True )
                    for i in manifest.edges.get( t, {} ).get( "inputs", [] ):
                        # after a crash, only what THIS graph makes: the units of the common graph
                        # are not ours to erase under another graph's lock
                        if i.endswith( ".o" ) and ( force_build() or not os.path.isabs( i ) or Path( i ).is_relative_to( self.root ) ):
                            ( self.root / i ).unlink( missing_ok = True )
            marker.write_text( f"pid { os.getpid() }\n" )

            # `LOOM_BUILD_JOBS`: the parallelism (default: ninja's, all cores) -- a CUDA
            # catalog compiles a hundred units of a gigabyte each, we do not want them all
            # at the same time on a shared machine
            jobs = env.var( "BUILD_JOBS" )
            cmd = [ ninja_path(), "-C", str( self.root ), "-f", str( ninja_file ),
                    *( [ "-j", jobs ] if jobs else [] ), *targets ]
            try:
                r = subprocess.run( cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT, text = True )
            finally:
                marker.unlink( missing_ok = True )          # only a death skips this
            out = r.stdout or ""
            # ninja's own line for a no-op build is noise; a real compilation is worth seeing
            worked = "no work to do" not in out
            if worked:
                print( out, end = "", flush = True )
            if r.returncode:
                raise RuntimeError( f"ninja failed ({ r.returncode }) on { targets }" )
            return worked


def verify_fast_path() -> bool:
    """`LOOM_VERIFY_FAST_PATH`: when the fast path says "up to date", run ninja anyway and fail if it
    finds work to do -- the check that the fast path never claims more than ninja would. For the
    test suite and for whoever touches the fingerprint."""
    return env.flag( "VERIFY_FAST_PATH" )


def _stat_key( path ):
    try:
        st = os.stat( path )
    except OSError:
        return None
    return [ st.st_mtime_ns, st.st_size ]


def _ninja_closure( graph: "_Graph", outputs: list ):
    """`{ output: [ dependency files ] }` as ninja's dependency log knows them (`ninja -t deps`) --
    the exact header closure of each object, which only the compiler could tell. `None` if the log
    has no valid record for one of them (then there is nothing to fingerprint)."""
    r = subprocess.run( [ ninja_path(), "-C", str( graph.root ), "-f", str( graph.root / "build.ninja" ),
                          "-t", "deps", *[ str( o ) for o in outputs ] ],
                        stdout = subprocess.PIPE, stderr = subprocess.DEVNULL, text = True )
    result, current = {}, None
    for line in r.stdout.splitlines():
        if not line.strip():
            current = None
        elif not line.startswith( " " ):
            target, _, status = line.partition( ": " )
            if "(VALID)" not in status:
                return None
            current = result[ target ] = []
        elif current is not None:
            current.append( line.strip() )
    return result


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
        self.own = _Graph( work_dir, self.sig, commands, relative = True ) if work_dir is not None else self.common

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
                include_first: list = (), include_only: list = None ) -> Path:
        """The object of a source compiled with these `defines` -- one unit per (source, defines,
        compiler).

        `include_first`: `-I` roots searched BEFORE the common ones -- a kernel's own overlay of
        generated headers (see `generated_headers.py`), which must win over the shared tree.

        `include_only`: replaces the common `-I` roots altogether. A unit that must not depend on
        what else is registered -- the runtime, which only needs loom's own headers -- says so: the
        roots GROW during a process (a package registers its own after the first kernels), and a
        command line that changes is a rebuild, of the unit and then of every library linked with it.

        `shared` (the default): the unit has dependents, the same `.o` serves all the kernels that
        ask for it, so it lives in the COMMON graph. `shared = False`: the object belongs to nobody
        else (a kernel's generated source), it lives in the own directory and goes away with it."""
        src = Path( src ).resolve()
        defines = dict( defines or {} )
        extra_flags = list( extra_flags )
        graph = self.common if shared else self.own
        include_first = [ graph.rel( d ) for d in include_first ]
        roots = [ str( d ) for d in ( include_dirs() if include_only is None else include_only ) ]
        # the name says everything that makes the object: the roots it is compiled against too, so
        # that two checkouts (or two sets of registered packages) sharing this directory never share
        # an object -- and never fight over one, a command line that changes being a rebuild. The
        # source of an own unit is named relative to its directory, which moves (`KernelDir`).
        name = f"{ src.stem }_{ _short_hash( self.sig, graph.rel( src ), sorted( defines.items() ), extra_flags, *include_first, *roots ) }.o"
        obj = ( graph.root / "obj" / name ) if graph is self.common else ( graph.root / name )
        graph.add_edge(
            obj, self._rule( self.compiler.rule_for( src ) ), [ src ],
            includes = " ".join( f"-I { _ninja_escape( d ) }" for d in [ *include_first, *roots ] ),
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
        objects = [ self.object( src, include_only = [ loom_include_root() ] ) for src in runtime_sources() ]
        # named after its objects, which name their sources and roots: another checkout of loom has
        # its own runtime, and the two never rewrite each other's file
        return self.shared_library(
            self.common.root / "runtime" / self.compiler.library_file_name(
                f"loom_runtime_{ _short_hash( self.sig, *objects ) }" ),
            objects, shared = True )

    # ── run ──────────────────────────────────────────────────────────────────
    def run( self, targets: list ):
        """Builds `targets` (and what they depend on), and nothing else.

        Two phases, in this order: the COMMON one first -- it may refresh the runtime or a
        domain unit -- then the OWN one, which then sees a newer input and relinks. The
        common artifacts enter the own graph as plain input files, with no
        rule: no kernel directory can decide to rebuild the runtime.

        FAST PATH (kernels only): ninja is the authority, but starting it twice costs ~60 ms per
        kernel to learn "nothing to do". Once ninja has said so, a STAMP next to the library caches
        its answer: the declared graph (commands, inputs), and the ( mtime, size ) of the outputs of
        both graphs and of every file in their dependency closure -- the very list ninja keeps
        (`ninja -t deps`), so a header this kernel does not include, or the shared generated tree
        other calls rewrite all day, never enters. Everything equal: ninja is skipped. Anything
        different: ninja decides exactly as before, and its "nothing to do" writes the stamp again.
        So the fast path never rebuilds more than ninja, and says "up to date" only when every
        file ninja would have compared is exactly as it was when ninja last said so.
        `LOOM_VERIFY_FAST_PATH=1` runs ninja behind it and fails if the two disagree."""
        if self.own is self.common:
            self.common.run( targets )
            return
        stamp_path = self.own.root / "ninja" / "fast.stamp"
        declared = self._declared()
        if not force_build() and self._stamp_matches( stamp_path, declared ):
            if not verify_fast_path():
                return
            worked = self.common.run( list( self.common.edges ) ) | self.own.run( targets )
            if worked:
                raise RuntimeError( f"loom: the fast path said { targets } were up to date, ninja rebuilt them" )
            return
        worked = self.common.run( list( self.common.edges ) ) | self.own.run( targets )
        if not force_build():
            self._write_stamp( stamp_path, declared, targets )

    def _declared( self ) -> str:
        edges = { **self.common.edges, **self.own.edges }
        return hashlib.sha1( json.dumps( [ self.common.rules, self.own.rules, edges ],
                                         sort_keys = True, default = str ).encode() ).hexdigest()

    def _stamp_matches( self, path, declared ) -> bool:
        try:
            old = json.loads( path.read_text() )
        except ( OSError, ValueError ):
            return False
        if old.get( "declared" ) != declared or not old.get( "outputs" ):
            return False
        return all( k is not None and _stat_key( os.path.join( self.own.root, f ) ) == k
                    for files in ( old[ "outputs" ], old.get( "deps", {} ) ) for f, k in files.items() )

    def _write_stamp( self, path, declared, targets ):
        """Records the state ninja has just validated. Refused (silently: the next run asks ninja
        again) unless every dependency is OLDER than the output it feeds -- ninja's own criterion --
        so a header edited while we were compiling can never be recorded as seen."""
        outputs = [ *self.common.edges, *self.own.edges ]
        deps = {}
        base = self.own.root       # what the paths of the stamp are relative to (the own graph's, a no-op for absolute ones)
        for graph, outs in ( ( self.common, list( self.common.edges ) ), ( self.own, [ self.own.rel( t ) for t in targets ] ) ):
            # a target may itself have no depfile (a link): only objects carry a closure
            objs = [ str( o ) for o in [ *outs, *[ i for o in outs for i in graph.edges.get( o, {} ).get( "inputs", [] ) ] ]
                     if str( o ).endswith( ".o" ) and str( o ) in graph.edges ]
            if not objs:
                continue
            with _Lock( graph.root ):
                closure = _ninja_closure( graph, sorted( set( objs ) ) )
            if closure is None:
                return
            for obj, files in closure.items():
                obj_key = _stat_key( os.path.join( graph.root, obj ) )
                for f in files:
                    k = _stat_key( os.path.join( graph.root, f ) )
                    if k is None or obj_key is None or k[ 0 ] > obj_key[ 0 ]:
                        return
                    deps[ f ] = k
        keys = { o: _stat_key( os.path.join( base, o ) ) for o in outputs }
        if any( k is None for k in keys.values() ):
            return
        tmp = path.with_suffix( ".tmp" )
        tmp.write_text( json.dumps( { "declared": declared, "outputs": keys, "deps": deps } ) )
        tmp.replace( path )


def runtime_sources() -> list:
    """The sources of `libloom_runtime` (`loom/cpp/runtime/*.cpp`), from a checkout or a wheel."""
    dev_root = _dev_repo_root()
    if dev_root is not None:
        root = dev_root / "loom" / "cpp" / "runtime"
    else:
        root = Path( __file__ ).resolve().parents[ 1 ] / "_cpp" / "runtime"
    return sorted( root.glob( "*.cpp" ) )
