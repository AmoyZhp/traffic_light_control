import importlib
from typing import Callable, Dict, Union

EntryPoint = Union[Callable, str]


def load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


class Spec(object):
    """A specification for a particular instance.

    Args:
        id(str): The instance id
        entry_point(Union[Callable, str]): 
            It could be a function(e.g build_env(config:Dict)-> Env ) or entryoint of a class (e.g. module.name:Class)
    """
    def __init__(self, id: str, entry_point: EntryPoint) -> None:
        self._id = id
        self._entry_point = entry_point

    def make(self, **kwargs):
        if self._entry_point is None:
            raise LookupError("env {} entry point id None".format(self._id))

        if callable(self._entry_point):
            instance = self._entry_point(**kwargs)
        else:
            cls = load(self._entry_point)
            instance = cls(**kwargs)
        return instance


class Registry(object):
    """Register an env by ID.

    """
    def __init__(self) -> None:
        self.specs: Dict[str, Spec] = {}

    def register(self, id: str, entry_point: EntryPoint, **kwargs):
        if id in self.specs:
            raise ValueError("Cannot re-register env {}".format(id))
        self.specs[id] = Spec(id=id, entry_point=entry_point, **kwargs)

    def make(self, id: str, **kwargs):
        return self.specs[id].make(**kwargs)
