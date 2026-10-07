"""Skip a test, with a notification, when the driver in use lacks a capability."""

_WHAT = { "grad": "autodiff", "vmap": "vmap", "trace": "a tracing jit", "jax": "jax itself", "cpu": "the CPU device" }

def need( capability: str ):
    """Call at the top of a test body: skips the test (reported as SKIP) if the active driver cannot do `capability`.

    `capability` is "grad" (autodiff: `grad`/`vjp`), "vmap", "trace" (a `jit` that traces its
    arguments into tracers), "jax" (the test calls jax directly) or "cpu" (host code: the driver's
    device must be the CPU). Only the numpy driver ( forward-only, `jit` is the identity ) lacks them today. `errand` is imported lazily: this module may be imported
    by an interpreter that is not the one the entry runs in.
    """
    from loom.drivers import framework_defaults
    if capability == "cpu":
        if not framework_defaults.device().is_cpu:
            from errand import skip
            skip( f"needs the CPU device -- the driver runs on { framework_defaults.device() }", hint = "run with a CPU environment" )
        return
    framework = str( framework_defaults.framework() )
    # torch differentiates, maps and compiles, but a kernel call is a break in the graph: nothing is
    # traced into tracers
    if capability == "jax":         # the test calls `jax` directly: nothing to fall back on
        lacks = framework != "jax"
    else:
        lacks = capability == "trace" if framework == "torch" else framework in ( "numpy", "cupy" )
    if not lacks:
        return
    if capability not in _WHAT:
        raise ValueError( f"unknown capability {capability!r}" )
    from errand import skip
    if capability == "jax":
        skip( f"calls jax directly -- not available under the { framework } driver", hint = "run with --env jax" )
    if framework == "torch":
        skip( f"needs {_WHAT[ capability ]} -- the torch jit does not trace loom calls", hint = "run with --env jax" )
    skip( f"needs {_WHAT[ capability ]} -- the {framework} driver is forward-only", hint = "run with --env jax or torch" )

def need_autodiff():
    need( "grad" )
