#include "external_value.h"
#if __has_include( <fake/value.h> )
#  include <fake/value.h>
#else
#  define FAKE_VALUE 0
#endif
namespace loom_test { int external_value() { return FAKE_VALUE; } }
