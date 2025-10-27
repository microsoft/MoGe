from typing import *
import time
from pathlib import Path
from numbers import Number
from functools import wraps
import warnings
import math
import json
import os
import importlib
import importlib.util


def catch_exception(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"Exception in {fn.__name__}",  end='r')
            # print({', '.join(repr(arg) for arg in args)}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())})
            traceback.print_exc(chain=False)
            time.sleep(0.1)
            return None
    return wrapper


class CallbackOnException:
    def __init__(self, callback: Callable, exception: type):
        self.exception = exception
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, self.exception):
            self.callback()
            return True
        return False
    
def traverse_nested_dict_keys(d: Dict[str, Dict]) -> Generator[Tuple[str, ...], None, None]:
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_key in traverse_nested_dict_keys(v):
                yield (k, ) + sub_key
        else:
            yield (k, )


def get_nested_dict(d: Dict[str, Dict], keys: Tuple[str, ...], default: Any = None):
    for k in keys:
        d = d.get(k, default)
        if d is None:
            break
    return d

def set_nested_dict(d: Dict[str, Dict], keys: Tuple[str, ...], value: Any):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def key_average(list_of_dicts: list) -> Dict[str, Any]:
    """
    Returns a dictionary with the average value of each key in the input list of dictionaries.
    """
    _nested_dict_keys = set()
    for d in list_of_dicts:
        _nested_dict_keys.update(traverse_nested_dict_keys(d))
    _nested_dict_keys = sorted(_nested_dict_keys)
    result = {}
    for k in _nested_dict_keys:
        values = []
        for d in list_of_dicts:
            v = get_nested_dict(d, k)
            if v is not None and not math.isnan(v):
                values.append(v)
        avg = sum(values) / len(values) if values else float('nan')
        set_nested_dict(result, k, avg)
    return result


def flatten_nested_dict(d: Dict[str, Any], parent_key: Tuple[str, ...] = None) -> Dict[Tuple[str, ...], Any]:
    """
    Flattens a nested dictionary into a single-level dictionary, with keys as tuples.
    """
    items = []
    if parent_key is None:
        parent_key = ()
    for k, v in d.items():
        new_key = parent_key + (k, )
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflattens a single-level dictionary into a nested dictionary, with keys as tuples.
    """
    result = {}
    for k, v in d.items():
        sub_dict = result
        for k_ in k[:-1]:
            if k_ not in sub_dict:
                sub_dict[k_] = {}
            sub_dict = sub_dict[k_]
        sub_dict[k[-1]] = v
    return result


def read_jsonl(file):
    import json
    with open(file, 'r') as f:
        data = f.readlines()
    return [json.loads(line) for line in data]


def write_jsonl(data: List[dict], file):
    import json
    with open(file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def to_hierachical_dataframe(data: List[Dict[Tuple[str, ...], Any]]):
    import pandas as pd
    data = [flatten_nested_dict(d) for d in data]
    df = pd.DataFrame(data)
    df = df.sort_index(axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)  
    return df


def recursive_replace(d: Union[List, Dict, str], mapping: Dict[str, str]):
    if isinstance(d, str):
        for old, new in mapping.items():
            d = d.replace(old, new)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            d[i] = recursive_replace(item, mapping)
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = recursive_replace(v, mapping)
    return d


class timeit:
    _history: Dict[str, List['timeit']] = {}

    def __init__(self, name: str = None, verbose: bool = True, average: bool = False):
        self.name = name
        self.verbose = verbose
        self.start = None
        self.end = None
        self.average = average
        if average and name not in timeit._history:
            timeit._history[name] = []

    def __call__(self, func: Callable):
        import inspect
        if inspect.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                with timeit(self.name or func.__qualname__):
                    ret = await func(*args, **kwargs)
                return ret
            return wrapper
        else:
            def wrapper(*args, **kwargs):
                with timeit(self.name or func.__qualname__):
                    ret = func(*args, **kwargs)
                return ret
            return wrapper
        
    def __enter__(self):
        self.start = time.time()
        return self

    @property
    def time(self) -> float:
        assert self.start is not None, "Time not yet started."
        assert self.end is not None, "Time not yet ended."
        return self.end - self.start

    @property
    def average_time(self) -> float:
        assert self.average, "Average time not available."
        return sum(t.time for t in timeit._history[self.name]) / len(timeit._history[self.name])

    @property
    def history(self) -> List['timeit']:
        return timeit._history.get(self.name, [])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        if self.average:
            timeit._history[self.name].append(self)
        if self.verbose:
            if self.average:
                avg = self.average_time
                print(f"{self.name or 'It'} took {avg:.6f} seconds in average.")
            else:
                print(f"{self.name or 'It'} took {self.time:.6f} seconds.")


def strip_common_prefix_suffix(strings: List[str]) -> List[str]:
    first = strings[0]

    for start in range(len(first)):
        if any(s[start] != strings[0][start] for s in strings):
            break

    for end in range(1, min(len(s) for s in strings)):
        if any(s[-end] != first[-end] for s in strings):
            break

    return [s[start:len(s) - end + 1] for s in strings]


def multithead_execute(inputs: List[Any], num_workers: int, pbar = None):
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import nullcontext
    from tqdm import tqdm

    if pbar is not None:
        pbar.total = len(inputs) if hasattr(inputs, '__len__') else None
    else:
        pbar = tqdm(total=len(inputs) if hasattr(inputs, '__len__') else None)

    def decorator(fn: Callable):
        with (
            ThreadPoolExecutor(max_workers=num_workers) as executor,
            pbar
        ):  
            pbar.refresh()
            @catch_exception
            @suppress_traceback
            def _fn(input):
                ret = fn(input)
                pbar.update()
                return ret
            executor.map(_fn, inputs)
            executor.shutdown(wait=True)
    
    return decorator


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


def import_file_as_module(file_path: Union[str, os.PathLike], module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module