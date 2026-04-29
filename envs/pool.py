import inspect

class EnvPool:
    """Yields a brand-new environment per session (thread-safe)."""
    def __init__(self, env_template):
        self._env_template = env_template

    class _Ctx:
        def __init__(self, template):
            cls = template.__class__

            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.keys())[1:]  # skip 'self'

            kwargs = {
                name: getattr(template, name)
                for name in params
                if hasattr(template, name)
            }

            self._env = cls(**kwargs)

        def __enter__(self):
            return self._env

        def __exit__(self, exc_type, exc, tb):
            return False

    def session(self):
        return EnvPool._Ctx(self._env_template)
