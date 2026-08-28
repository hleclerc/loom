"""Le CLI (`./run`). Les entrées -- test, bench, experiment -- sont déclarées dans
`loom.testing`, pas ici ; ce module les ré-exporte pour que les fichiers écrits contre
l'ancien harnais (`from loom.cli import experiment, Param`) continuent de marcher.
"""
from loom.testing import Param, Args, test, bench, experiment
