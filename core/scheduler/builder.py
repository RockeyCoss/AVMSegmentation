from core.utils import Registry

SCHEDULER = Registry('scheduler')


def build_scheduler(config: dict):
    return SCHEDULER.build(config)
