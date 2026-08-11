"""Test ComputedAttribute caching and invalidation."""

from sdot.util.Aggregate import Aggregate
from sdot.util.ComputedAttribute import ComputedAttribute
from sdot.tensor.Tensor import Tensor


class SimpleBox( Aggregate ):
    """Simple test aggregate with a computed area.

    The area field depends on width and height; when either changes, area is invalidated.
    """

    width  : Tensor
    height : Tensor
    area   : ComputedAttribute[ Tensor, ("width", "height") ]


def test_computed_attribute_invalidation():
    """Test that ComputedAttribute is invalidated when dependencies change."""
    box = SimpleBox()

    import numpy as np
    box.width = np.array( 5.0 )
    box.height = np.array( 3.0 )

    # Check that area is valid initially
    assert box._computed_attrs["area"]._cache_valid, "Cache should be valid initially"

    # Modify width (should invalidate)
    box.width = np.array( 4.0 )
    assert not box._computed_attrs["area"]._cache_valid, "Cache should be invalidated after width change"

    # Manually mark as valid (e.g., after computing)
    box._computed_attrs["area"]._cache_valid = True
    assert box._computed_attrs["area"]._cache_valid, "Cache should be valid after manual update"


def test_computed_attribute_multiple_dependencies():
    """Test that ComputedAttribute invalidates when ANY dependency changes."""
    box = SimpleBox()

    import numpy as np
    box.width = np.array( 2.0 )
    box.height = np.array( 3.0 )

    box._computed_attrs["area"]._cache_valid = True

    # Modify height (should also invalidate)
    box.height = np.array( 4.0 )
    assert not box._computed_attrs["area"]._cache_valid, "Cache should be invalidated when any dependency changes"


if __name__ == "__main__":
    test_computed_attribute_caching()
    test_computed_attribute_multiple_dependencies()
    print("All tests passed!")
