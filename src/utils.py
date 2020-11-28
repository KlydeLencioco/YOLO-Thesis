import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, Tuple

config = ConfigParser()
current_path = Path(__file__).parent.absolute()
config.read(os.path.join(current_path, "config.ini"))


def get_config_kwargs(section: str, kwargs: Tuple[Tuple[str, Any]]) -> Dict[str, Any]:
    section_kwargs = {}

    try:
        values = config[section]
    except Exception:
        return section_kwargs

    for kwarg, convert in kwargs:
        value = values.get(kwarg)
        if value:
            section_kwargs[kwarg] = convert(value)

    return section_kwargs


def get_config(section: str, kwarg: str, convert: Any, default: Any) -> Any:
    try:
        values = config[section]
    except Exception:
        return default

    value = values.get(kwarg)
    if value:
        return convert(value)

    return default
