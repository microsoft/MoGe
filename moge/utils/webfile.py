from typing import *

import requests

__all__ = ["WebFile"]


class WebFile:  
    def __init__(self, url: str, session: Optional[requests.Session] = None, headers: Optional[Dict[str, str]] = None, size: Optional[int] = None):  
        self.url = url  
        self.session = session or requests.Session()  
        self.session.headers.update(headers or {})
        self._offset = 0  
        self.size = size if size is not None else self._fetch_size()
  
    def _fetch_size(self):  
        with self.session.get(self.url, stream=True) as response:  
            response.raise_for_status()  
            content_length = response.headers.get("Content-Length")  
            if content_length is None:  
                raise ValueError("Missing Content-Length in header")  
            return int(content_length) 

    def _fetch_data(self, offset: int, n: int) -> bytes:
        headers = {"Range": f"bytes={offset}-{min(offset + n - 1, self.size)}"}
        response = self.session.get(self.url, headers=headers)
        response.raise_for_status()
        return response.content
  
    def seekable(self) -> bool:  
        return True  
  
    def tell(self) -> int:  
        return self._offset  
  
    def available(self) -> int:  
        return self.size - self._offset  
  
    def seek(self, offset: int, whence: int = 0) -> None:  
        if whence == 0:  
            new_offset = offset  
        elif whence == 1:  
            new_offset = self._offset + offset  
        elif whence == 2:  
            new_offset = self.size + offset  
        else:  
            raise ValueError("Invalid value for whence")  

        self._offset = max(0, min(new_offset, self.size))  
  
    def read(self, n: Optional[int] = None) -> bytes:  
        if n is None or n < 0:  
            n = self.available()  
        else:  
            n = min(n, self.available())  

        if n == 0:  
            return b''  

        data = self._fetch_data(self._offset, n)
        self._offset += len(data)  
  
        return data  

    def close(self) -> None:  
        pass

    def __enter__(self):  
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
  
    