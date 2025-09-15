import os
from pathlib import Path

def pytest_sessionstart(session):
    plugin_init = Path(__file__).parent / "__init__.py"
    if plugin_init.exists():
        os.rename(plugin_init, plugin_init.with_suffix(".py.bak"))
        session.config._plugin_init_backup = plugin_init

def pytest_sessionfinish(session, exitstatus):
    plugin_init = getattr(session.config, "_plugin_init_backup", None)
    if plugin_init and plugin_init.with_suffix(".py.bak").exists():
        os.rename(plugin_init.with_suffix(".py.bak"), plugin_init)