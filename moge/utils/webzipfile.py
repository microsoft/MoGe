import struct
from typing import *
from zipfile import (
    _FH_EXTRA_FIELD_LENGTH,
    _FH_FILENAME_LENGTH,
    _FH_GENERAL_PURPOSE_FLAG_BITS,
    _FH_SIGNATURE,
    _MASK_COMPRESSED_PATCH,
    _MASK_ENCRYPTED,
    _MASK_STRONG_ENCRYPTION,
    _MASK_UTF_FILENAME,
    BadZipFile,
    ZipExtFile,
    ZipFile,
    ZipInfo,
    sizeFileHeader,
    stringFileHeader,
    structFileHeader,
)

from requests import Session

from .webfile import WebFile


class _SharedWebFile(WebFile):
    def __init__(self, webfile: WebFile, pos: int):
        super().__init__(webfile.url, webfile.session, size=webfile.size)
        self.seek(pos)


class WebZipFile(ZipFile):
    "Lock-free version of ZipFile that reads from a WebFile, allowing for concurrent reads."
    def __init__(self, url: str, session: Optional[Session] = None, headers: Optional[Dict[str, str]] = None):
        """Open the ZIP file with mode read 'r', write 'w', exclusive create 'x',
        or append 'a'."""
        webf = WebFile(url, session=session, headers=headers)
        super().__init__(webf, mode='r')

    def open(self, name, mode="r", pwd=None, *, force_zip64=False):
        """Return file-like object for 'name'.

        name is a string for the file name within the ZIP file, or a ZipInfo
        object.

        mode should be 'r' to read a file already in the ZIP file, or 'w' to
        write to a file newly added to the archive.

        pwd is the password to decrypt files (only used for reading).

        When writing, if the file size is not known in advance but may exceed
        2 GiB, pass force_zip64 to use the ZIP64 format, which can handle large
        files.  If the size is known in advance, it is best to pass a ZipInfo
        instance for name, with zinfo.file_size set.
        """
        if mode not in {"r", "w"}:
            raise ValueError('open() requires mode "r" or "w"')
        if pwd and (mode == "w"):
            raise ValueError("pwd is only supported for reading files")
        if not self.fp:
            raise ValueError(
                "Attempt to use ZIP archive that was already closed")
        
        assert mode == "r", "Only read mode is supported for now"

        # Make sure we have an info object
        if isinstance(name, ZipInfo):
            # 'name' is already an info object
            zinfo = name
        elif mode == 'w':
            zinfo = ZipInfo(name)
            zinfo.compress_type = self.compression
            zinfo._compresslevel = self.compresslevel
        else:
            # Get info object for name
            zinfo = self.getinfo(name)

        if mode == 'w':
            return self._open_to_write(zinfo, force_zip64=force_zip64)

        if self._writing:
            raise ValueError("Can't read from the ZIP file while there "
                    "is an open writing handle on it. "
                    "Close the writing handle before trying to read.")

        # Open for reading:
        self._fileRefCnt += 1
        zef_file = _SharedWebFile(self.fp, zinfo.header_offset)

        try:
            # Skip the file header:
            fheader = zef_file.read(sizeFileHeader)
            if len(fheader) != sizeFileHeader:
                raise BadZipFile("Truncated file header")
            fheader = struct.unpack(structFileHeader, fheader)
            if fheader[_FH_SIGNATURE] != stringFileHeader:
                raise BadZipFile("Bad magic number for file header")

            fname = zef_file.read(fheader[_FH_FILENAME_LENGTH])
            if fheader[_FH_EXTRA_FIELD_LENGTH]:
                zef_file.seek(fheader[_FH_EXTRA_FIELD_LENGTH], whence=1)

            if zinfo.flag_bits & _MASK_COMPRESSED_PATCH:
                # Zip 2.7: compressed patched data
                raise NotImplementedError("compressed patched data (flag bit 5)")

            if zinfo.flag_bits & _MASK_STRONG_ENCRYPTION:
                # strong encryption
                raise NotImplementedError("strong encryption (flag bit 6)")

            if fheader[_FH_GENERAL_PURPOSE_FLAG_BITS] & _MASK_UTF_FILENAME:
                # UTF-8 filename
                fname_str = fname.decode("utf-8")
            else:
                fname_str = fname.decode(self.metadata_encoding or "cp437")

            if fname_str != zinfo.orig_filename:
                raise BadZipFile(
                    'File name in directory %r and header %r differ.'
                    % (zinfo.orig_filename, fname))

            # check for encrypted flag & handle password
            is_encrypted = zinfo.flag_bits & _MASK_ENCRYPTED
            if is_encrypted:
                if not pwd:
                    pwd = self.pwd
                if pwd and not isinstance(pwd, bytes):
                    raise TypeError("pwd: expected bytes, got %s" % type(pwd).__name__)
                if not pwd:
                    raise RuntimeError("File %r is encrypted, password "
                                       "required for extraction" % name)
            else:
                pwd = None

            return ZipExtFile(zef_file, mode, zinfo, pwd, True)
        except:
            zef_file.close()
            raise