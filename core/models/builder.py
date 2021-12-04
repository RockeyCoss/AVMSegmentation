from core.utils import Registry

MODELS = Registry('models')


def build_model(config: dict):
    return MODELS.build(config)
