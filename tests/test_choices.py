"""A kernel runs as the architecture its table entry says -- and as the real one when there is none (no fallback)."""
from pathlib import Path
import tempfile

from loom import choices
from loom.devices import hardware
from loom.devices.Cpu import Cpu
from errand import test
from test_catalogue_targets import _x86, X86


def _fresh():
    choices._roots.clear(); choices._cache.clear()
    d = Path( tempfile.mkdtemp() )
    choices.register_choices( d )
    return d


if test( "without_an_entry_a_kernel_runs_as_the_real_architecture" ):
    _fresh()
    with _x86( "x86-64-v4" ):
        kp = choices.kernel_parameters_for( "k", device = Cpu() )
        assert kp.target == "x86-64-v4" and kp.origin == "real"


if test( "an_entry_makes_the_kernel_and_its_device_behave_as_a_poorer_architecture" ):
    root = _fresh()
    with _x86( "x86-64-v4" ):
        choices.record( root, "k", { "dim": 3 }, "x86-64-v4", "x86-64-v3", cpu = "test" )
        kp = choices.kernel_parameters_for( "k", device = Cpu(), dim = 3 )
        assert kp.target == "x86-64-v3" and kp.origin == "table"
        assert hardware.mask() is None                            # nothing leaks out of the scope
        with kp.scope():
            assert kp.device.catalogue_target() == "x86-64-v3"
            assert kp.device.current_catalogue_tag() == "cpu-x86-64-v3"
        assert kp.device.catalogue_target() == "x86-64-v4"
        # other extra keys, other kernel: no entry, so the real one
        assert choices.kernel_parameters_for( "k", device = Cpu(), dim = 2 ).target == "x86-64-v4"
        assert choices.kernel_parameters_for( "other", device = Cpu() ).target == "x86-64-v4"


if test( "an_entry_is_per_real_architecture_and_the_files_do_not_collide" ):
    root = _fresh()
    choices.record( root, "k", {}, "x86-64-v4", "x86-64-v3" )
    choices.record( root, "k", {}, "x86-64-v3", "x86-64-v2" )
    d = root / choices.directory_name( "k", {} )
    assert sorted( p.name for p in d.iterdir() ) == [ "params.json", "x86-64-v3.json", "x86-64-v4.json" ]
    with _x86( "x86-64-v3" ):
        assert choices.kernel_parameters_for( "k", device = Cpu() ).target == "x86-64-v2"


if test( "an_entry_stronger_than_the_machine_is_refused" ):
    root = _fresh()
    choices.record( root, "k", {}, "x86-64-v3", "x86-64-v4" )
    with _x86( "x86-64-v3" ):
        try:
            choices.kernel_parameters_for( "k", device = Cpu() )
            assert False
        except ValueError as e:
            assert "stronger" in str( e )


if test( "loom_target_is_above_the_table" ):
    root = _fresh()
    choices.record( root, "k", {}, "x86-64-v4", "x86-64-v3" )
    with _x86( "x86-64-v4", mask = "x86-64-v2" ):
        kp = choices.kernel_parameters_for( "k", device = Cpu() )
        assert kp.target == "x86-64-v2" and kp.origin == "mask"


if test( "a_missing_architecture_is_answered_by_its_closest_neighbour" ):
    root = _fresh()
    choices.record( root, "k", {}, "x86-64-v3", "x86-64-v2" )
    with _x86( "x86-64-v4" ):
        kp = choices.kernel_parameters_for( "k", device = Cpu() )     # no v4 file: the poorer level stands for it
        assert kp.target == "x86-64-v2" and kp.origin == "table:x86-64-v3"
    with _x86( "x86-64-v2" ):
        assert choices.kernel_parameters_for( "k", device = Cpu() ).origin == "real"   # a poorer machine is nobody's neighbour


if test( "a_neighbours_answer_is_clipped_to_what_the_machine_is" ):
    root = _fresh()
    choices.record( root, "k", {}, "x86-64-v4", "x86-64-v4" )
    with _x86( "x86-64-v3" ):
        assert choices.kernel_parameters_for( "k", device = Cpu() ).origin == "real"   # only poorer levels are neighbours
    assert hardware.neighbours( "sm_86" )[ :5 ] == [ "sm_86", "sm_80", "sm_89", "sm_75", "sm_70" ]
    assert hardware.neighbours( "sm_70" )[ 0 ] == "sm_70" and hardware.neighbours( "sm_75" )[ :2 ] == [ "sm_75", "sm_70" ]
    assert hardware.clip( "sm_89", "sm_86" ) == "sm_86" and hardware.clip( "sm_80", "sm_86" ) == "sm_80"


if test( "the_benchmark_retains_the_poorest_architecture_within_tolerance" ):
    from loom import benchmark
    t = { "x86-64-v4": 1.00, "x86-64-v3": 1.05, "x86-64-v2": 1.60, "x86-64": 2.0 }
    assert benchmark.retain( t, 0.1 ) == "x86-64-v3"
    assert benchmark.retain( t, 0.0 ) == "x86-64-v4"
    assert benchmark.retain( t, 5.0 ) == "x86-64"


if test( "measurements_gathered_in_the_loop_record_the_choice_of_the_real_architecture" ):
    from loom import benchmark
    import time
    root = _fresh()
    with _x86( "x86-64-v4" ):
        m = benchmark.Measurements( root, tolerance = 10.0, repeats = 2 )
        for target in ( "x86-64-v4", "x86-64-v3" ):
            m.time( "k", lambda: time.sleep( 0.002 ), hardware.Run( "cpu", target ), dim = 2 )
        done = m.write()
        assert [ ( d[ 0 ], d[ 3 ], d[ 4 ] ) for d in done ] == [ ( "k", "x86-64-v4", "x86-64-v3" ), ( "k", "x86-64-v3", "x86-64-v3" ) ]
        f = root / choices.directory_name( "k", { "dim": 2 } ) / "x86-64-v4.json"
        assert f.is_file() and '"use": "x86-64-v3"' in f.read_text()


if test( "prune_removes_the_kernels_no_machine_will_ask_for" ):
    from loom.compilation import catalogue
    import json
    choices_dir = _fresh()
    for target, use in ( ( "x86-64-v4", "x86-64-v2" ), ( "x86-64-v3", "x86-64-v2" ), ( "x86-64-v2", "x86-64-v2" ), ( "x86-64", "x86-64" ) ):
        choices.record( choices_dir, "k", {}, target, use )
    owner = { "name": "k", "extra_keys": {} }
    with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as src, _x86( "x86-64-v4" ):
        lib = Path( src ) / "kernel.so"
        lib.write_bytes( b"binary" )
        import os
        os.environ[ "LOOM_CACHE_DIR" ] = src
        from test_catalogue_targets import _publish_env
        with _publish_env( src ), catalogue.collect( out ):
            for target in X86:
                with _x86( "x86-64-v4", mask = target ):
                    with catalogue.owned_by( owner ):
                        catalogue.publish( lib, "managed", [], Cpu(), "managed", "sig" )
                    catalogue.publish( lib, "free", [], Cpu(), "free", "sig" )               # asked for by nobody in particular
        gone = catalogue.prune( out, [ choices_dir ] )
        kept = {k: sorted( v ) for k, v in catalogue.index( out ).items()}
        managed, free = catalogue.key( "managed", [] ), catalogue.key( "free", [] )
        assert sorted( t for t, k in gone ) == [ "cpu-x86-64-v3", "cpu-x86-64-v4" ] and all( k == managed for _, k in gone )
        assert kept[ "cpu-x86-64-v2" ] == sorted( [ managed, free ] ) and kept[ "cpu-x86-64-v4" ] == [ free ]
        assert not ( Path( out ) / "cpu-x86-64-v4" / ( managed + catalogue.library_suffix() ) ).exists()


if test( "a_real_measurement_is_not_overwritten_by_an_emulated_one" ):
    root = _fresh()
    choices.record( root, "k", {}, "x86-64-v3", "x86-64-v3", cpu = "real v3" )
    choices.record( root, "k", {}, "x86-64-v3", "x86-64-v2", emulated_on = "x86-64-v4" )
    assert choices.read( "k", {}, "x86-64-v3" ) == ( "x86-64-v3", "x86-64-v3" )
    choices.record( root, "k", {}, "x86-64-v2", "x86-64", emulated_on = "x86-64-v4" )
    choices.record( root, "k", {}, "x86-64-v2", "x86-64-v2", emulated_on = "x86-64-v4" )       # an emulated one is replaced by a newer
    assert choices.read( "k", {}, "x86-64-v2" ) == ( "x86-64-v2", "x86-64-v2" )


if test( "bench_loop_is_a_plain_loop_outside_a_build_and_a_timer_inside_one" ):
    from loom import benchmark
    import time
    calls = []
    for _ in benchmark.bench_loop( "k" ):
        calls.append( 1 )
    assert len( calls ) == 1                                   # the usage stays a usage
    root = _fresh()
    with _x86( "x86-64-v4" ), benchmark.Measurements( root ) as m:
        for target in ( "x86-64-v4", "x86-64-v3" ):
            with hardware.behave_as( hardware.Run( "cpu", target ) ):
                for _ in benchmark.bench_loop( "k", weight = 2.0, repeats = 2, min_seconds = 0, dim = 3 ):
                    calls.append( 1 )
                    time.sleep( 0.002 )
                for _ in benchmark.bench_loop( "k", weight = 1.0, repeats = 2, min_seconds = 0, dim = 3 ):      # the same kernel: they add up
                    time.sleep( 0.002 )
        assert len( calls ) == 1 + 2 * 3                         # a warm-up and two timed runs, on each architecture
        (name, keys, device, target, use, timings), *_ = m.decisions()
        assert name == "k" and keys == { "dim": 3 } and target == "x86-64-v4" and set( timings ) == { "x86-64-v4", "x86-64-v3" }
        assert len( m.cases[ ( "k", choices._canonical( { "dim": 3 } ), "cpu" ) ] ) == 2          # two experiments on the same kernel
    assert benchmark._active is None


if test( "experiments_on_one_kernel_add_their_ratios_with_their_weights" ):
    from loom import benchmark
    m = benchmark.Measurements( "unused", tolerance = 0.05 )
    v4, v3 = hardware.Run( "cpu", "x86-64-v4" ), hardware.Run( "cpu", "x86-64-v3" )
    # experiment 0: v4 is 40 % faster ; experiment 1 (long, other use): v4 gains nothing. Same weight: mean ratio of v3 = 1.2
    m.add( "k", {}, v4, 0.6, 1.0 ); m.add( "k", {}, v4, 100.0, 1.0 )
    m.add( "k", {}, v3, 1.0, 1.0 ); m.add( "k", {}, v3, 100.0, 1.0 )
    k = ( "k", "{}", "cpu" )
    sc = m.scores( k, [ "x86-64-v4", "x86-64-v3" ] )
    assert abs( sc[ "x86-64-v4" ] - 1.0 ) < 1e-9 and abs( sc[ "x86-64-v3" ] - ( 1.0 / 0.6 + 1.0 ) / 2 ) < 1e-9
    assert benchmark.retain( sc, 0.05 ) == "x86-64-v4"
    # the weight of the first experiment drops: v3 is within 5 % on the weighted mean
    m2 = benchmark.Measurements( "unused", tolerance = 0.05 )
    m2.add( "k", {}, v4, 0.6, 0.05 ); m2.add( "k", {}, v4, 100.0, 1.0 )
    m2.add( "k", {}, v3, 1.0, 0.05 ); m2.add( "k", {}, v3, 100.0, 1.0 )
    assert benchmark.retain( m2.scores( k, [ "x86-64-v4", "x86-64-v3" ] ), 0.05 ) == "x86-64-v3"
