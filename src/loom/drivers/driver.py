from typing import TYPE_CHECKING, cast
from .DriverProxy import DriverProxy

# start with an unknown driver instance
driver = DriverProxy()

# give a concrete instance to typing
if TYPE_CHECKING:
    from .JaxDriver import JaxDriver
    driver = cast( JaxDriver, driver )
