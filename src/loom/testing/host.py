def host( x ):
    """What a value holds, as a numpy array on the host -- whichever framework made it and wherever it
    lives. `numpy.asarray( x )` does that for numpy and Jax; a torch tensor on a card refuses it (the
    copy back is not implicit), which this does not."""
    from loom.drivers import framework_defaults
    return framework_defaults.ops( x ).to_numpy( x )
