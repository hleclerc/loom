"""Where loom runs, one environment per framework:

    errand run --env jax,numpy,torch tests

`errand-vars.py` ( not versioned, read first ) adds the machine with a GPU, if there is one:

    # errand-vars.py
    mamba_root  = "~/.mamba"                  # where the environments go
    gpu_host    = "a machine with an NVIDIA card ( driver >= 580 )"
    gpu_root    = "a directory over there"
    gpu_mamba   = "its micromamba root"
"""
import errand

import os

HERE = os.path.dirname( os.path.abspath( __file__ ) )
PIP  = [ "numpy", "typing_extensions", "pyyaml", "devtools", "ninja", "pytest", "matplotlib" ]


def framework_env( framework, *, host = None, root = HERE, mamba_root = None ):
    """`framework` here, or on `host` ( with a GPU: the CUDA stack comes from pip, and loom's pinned nvcc with `loom[cuda]` )."""
    cuda = host is not None
    pip  = [ *PIP,
             *( [ "jax[cuda13]" ] if cuda and framework == "jax" else [ ] if framework == "numpy" else [ framework ] ),
             "-e", f"{ root }[cuda]" if cuda else root ]
    return errand.Env( [
        *( [ errand.Ssh( host = host, root = root ) ] if host else [ ] ),
        errand.Micromamba( f"loom-{ framework }", python = "3.13", pip = pip, root_prefix = mamba_root,
                           # the derivative kernels need P2280, which Apple's clang does not have
                           packages = [ ] if cuda else [ "clangxx" ] ),
        errand.Vars( { "LOOM_FRAMEWORK": framework, **( { } if cuda else { "LOOM_CXX": "clang++", "LOOM_CXXFLAGS": "-std=c++23" } ) } ),
    ], driver = framework, **( { "cuda": True } if cuda else { } ) )


for framework in ( "jax", "numpy", "torch" ):
    errand.envs[ framework ] = framework_env( framework, mamba_root = errand.value( "mamba_root" ) )

if errand.value( "gpu_host" ):
    for framework in ( "jax", "numpy", "torch", "cupy" ):
        errand.envs[ f"gpu-{ framework }" ] = framework_env( framework, host = errand.value( "gpu_host" ), root = errand.value( "gpu_root" ),
                                                              mamba_root = errand.value( "gpu_mamba" ) )
