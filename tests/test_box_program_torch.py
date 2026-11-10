"""`loom.BoxProgram` on 4 Torch processes ( gloo ): a script of its own, `mp.spawn` re-imports its main module."""
import sys

from errand import has_tag, test

if not has_tag( "driver=torch" ):
    sys.exit( 0 )

if test( "BoxProgram on Torch processes: spread blocks or whole tensors, what a numpy loop gives" ):
    import os
    import subprocess
    script = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "box_program_torch_worker.py" )
    r = subprocess.run( [ sys.executable, script ], capture_output = True, text = True, timeout = 900 )
    assert r.returncode == 0 and "all ranks ok" in r.stdout, ( r.stdout[ -2000: ], r.stderr[ -4000: ] )
