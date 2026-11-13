#pragma once
// A unit that includes an EXTERNAL library ( `tests/test_externals.py` ): `fake/value.h` is on its
// include path only when the kernel that links it asks for the external ( `FfiCode( externals = ... )` ).
#include <loom/support/export.h>
namespace loom_test { LOOM_EXPORT int external_value(); }
