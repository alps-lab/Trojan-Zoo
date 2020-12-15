# -*- coding: utf-8 -*-

from trojanzoo.utils.param import Param

import os
import json
import yaml


class Config:
    """A singleton class to process config. The config is composed of ``package``, ``user``, ``project`` and ``cmd``.

    Attributes:
        package (Param): The global config saved in ``trojanzoo/config/``
        user (Param): The user config
        project (Param): The project config saved in ``./config/``
        cmd (Param): The config from ``path``, usually passed by ``--config`` in command line.

        config (Param): The combined config.
    """

    package = Param()
    user = Param()
    project = Param()
    cmd = Param()

    config = Param()

    path = {
        'package': os.path.dirname(__file__),
        'user': None,
        'project': './config/',
        'cmd': None
    }

    @classmethod
    def refresh_config(cls) -> Param:
        cls.config = Param()
        for element in [cls.package, cls.user, cls.project, cls.cmd]:
            cls.config.update(element)
        return cls.config

    @staticmethod
    def load_config(path: str) -> dict:
        if path is None:
            return {}
        if not isinstance(path, str):
            raise TypeError(path)
        if os.path.isdir(path):
            if path[-1] != '/' and path[-1] != '\\':
                path += '/'
            _dict = {}
            for root, dirs, files in os.walk(path):
                for _file in files:
                    name, ext = os.path.splitext(_file)
                    if ext in ['.yml', '.yaml', 'json']:
                        _dict |= {name: Config.load_config(os.path.join(root, _file))}
            return _dict
        elif os.path.isfile(path):
            name, ext = os.path.splitext(os.path.split(path)[1])
            if ext in ['.yml', 'yaml']:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.load(f.read(), Loader=yaml.FullLoader)
            elif ext == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f.read())
            else:
                return {}
        else:
            return {}

    @classmethod
    def update(cls, item_list: list[str] = ['package', 'user', 'project', 'cmd']):
        """Update the config

        Args:
            args (list[str]): values in ``['package', 'user', 'project']``
        """
        for item in item_list:
            setattr(cls, item, Param(cls.load_config(cls.path[item])))
        cls.refresh_config()

    @classmethod
    def set_config_path(cls, item: str, loc: str):
        cls.path[item] = loc

    @staticmethod
    def combine_param(config: str = None, dataset_name: str = None, **kwargs) -> Param:
        r"""Combine parser arguments and config parameters. The values in config are picked according to ``dataset``.

        Args:
            config (Param): config parameters
            dataset_name (str): dataset used to pick values in config. Default: None.

        Returns:
            combined :class:`Param`.
        """
        result = Param()
        if config is not None:
            result.update(config)
        for key, value in result.items():
            if isinstance(value, Param):
                result[key] = value[dataset_name]
        result.update(kwargs)
        return result


Config.update()
config = Config.config