import inspect


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    def _registry_module(self, module_class, module_name, force=False):
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class')
        if module_name is None:
            module_name = module_class.__name__
        if module_name in self._module_dict and not force:
            raise KeyError(f'{module_name} is already registered '
                           f'in {self.name}')
        self._module_dict[module_name] = module_class

    def registry_module(self, name=None, force=False, module=None):
        if module is not None:
            self._registry_module(module_class=module,
                                  module_name=name,
                                  force=force)
            return module

        def _registry(cls):
            self._registry_module(module_class=cls,
                                  module_name=name,
                                  force=force)
            return cls

        return _registry

    def get(self, key):
        return self._module_dict.get(key)

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, item):
        return self._module_dict.get(item) is not None

    def __getitem__(self, item):
        return self._module_dict[item]

    def build(self, cfg):
        assert isinstance(cfg, dict)
        assert 'type' in cfg
        args = cfg.copy()
        type = args.pop('type')
        assert isinstance(type, str)
        obj_cls = self.get(type)
        if obj_cls is None:
            raise KeyError
        if not inspect.isclass(obj_cls):
            raise ValueError
        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')
