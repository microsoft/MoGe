from functools import wraps
import warnings


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise
    return wrapper


class no_warnings:
    def __init__(self, action: str = 'ignore', **kwargs):
        self.action = action
        self.filter_kwargs = kwargs
    
    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(self.action, **self.filter_kwargs)
                return fn(*args, **kwargs)
        return wrapper  
    
    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter(self.action, **self.filter_kwargs)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warnings_manager.__exit__(exc_type, exc_val, exc_tb)
