from hprl.registration import Registry, EntryPoint

registry = Registry()


def register(id: str, entry_point: EntryPoint, **kwargs):
    registry.register(id=id, entry_point=entry_point, **kwargs)


def make(id: str, **kwargs):
    return registry.make(id=id, **kwargs)