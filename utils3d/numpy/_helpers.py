# decorator
import numpy as np
from numbers import Number
import inspect
from functools import wraps
from typing import *
from .._helpers import suppress_traceback


def get_args_order(func, args, kwargs):
    """
    Get the order of the arguments of a function.
    """
    names = inspect.getfullargspec(func).args
    names_idx = {name: i for i, name in enumerate(names)}
    args_order = []
    kwargs_order = {}
    for name, arg in kwargs.items():
        if name in names:
            kwargs_order[name] = names_idx[name]
            names.remove(name)
    for i, arg in enumerate(args):
        if i < len(names):
            args_order.append(names_idx[names[i]])
    return args_order, kwargs_order


def broadcast_args(args, kwargs, args_dim, kwargs_dim):
    spatial = []
    for arg, arg_dim in zip(args + list(kwargs.values()), args_dim + list(kwargs_dim.values())):
        if isinstance(arg, np.ndarray) and arg_dim is not None:
            arg_spatial = arg.shape[:arg.ndim-arg_dim]
            if len(arg_spatial) > len(spatial):
                spatial = [1] * (len(arg_spatial) - len(spatial)) + spatial
            for j in range(len(arg_spatial)):
                if spatial[-j] < arg_spatial[-j]:
                    if spatial[-j] == 1:
                        spatial[-j] = arg_spatial[-j]
                    else:
                        raise ValueError("Cannot broadcast arguments.")
    for i, arg in enumerate(args):
        if isinstance(arg, np.ndarray) and args_dim[i] is not None:
            args[i] = np.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-args_dim[i]:]])
    for key, arg in kwargs.items():
        if isinstance(arg, np.ndarray) and kwargs_dim[key] is not None:
            kwargs[key] = np.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-kwargs_dim[key]:]])
    return args, kwargs, spatial


def batched(*dims):
    """
    Decorator that allows a function to be called with batched arguments.
    """
    def decorator(func):
        @wraps(func)
        @suppress_traceback
        def wrapper(*args, **kwargs):
            args = list(args)
            # get arguments dimensions
            args_order, kwargs_order = get_args_order(func, args, kwargs)
            args_dim = [dims[i] for i in args_order]
            kwargs_dim = {key: dims[i] for key, i in kwargs_order.items()}
            # convert to numpy array
            for i, arg in enumerate(args):
                if isinstance(arg, (Number, list, tuple)) and args_dim[i] is not None:
                    args[i] = np.array(arg)
            for key, arg in kwargs.items():
                if isinstance(arg, (Number, list, tuple)) and kwargs_dim[key] is not None:
                    kwargs[key] = np.array(arg)
            # broadcast arguments
            args, kwargs, spatial = broadcast_args(args, kwargs, args_dim, kwargs_dim)
            for i, (arg, arg_dim) in enumerate(zip(args, args_dim)):
                if isinstance(arg, np.ndarray) and arg_dim is not None:
                    args[i] = arg.reshape([-1, *arg.shape[arg.ndim-arg_dim:]])
            for key, arg in kwargs.items():
                if isinstance(arg, np.ndarray) and kwargs_dim[key] is not None:
                    kwargs[key] = arg.reshape([-1, *arg.shape[arg.ndim-kwargs_dim[key]:]])
            # call function
            results = func(*args, **kwargs)
            type_results = type(results)
            results = list(results) if isinstance(results, (tuple, list)) else [results]
            # restore spatial dimensions
            for i, result in enumerate(results):
                results[i] = result.reshape([*spatial, *result.shape[1:]])
            if type_results == tuple:
                results = tuple(results)
            elif type_results == list:
                results = list(results)
            else:
                results = results[0]
            return results
        return wrapper
    return decorator
