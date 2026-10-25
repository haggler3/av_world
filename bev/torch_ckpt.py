"""Read a PyTorch (zip) checkpoint's tensors without torch and without running any code from the file: the pickle is read
with a whitelist that maps torch's tensor-rebuild functions to inert descriptors, and every other global to a placeholder
string, so a malicious checkpoint cannot execute anything."""
import collections
import pickle
import zipfile

import numpy as np

_DT = {"FloatStorage": "float32", "HalfStorage": "float16", "LongStorage": "int64", "IntStorage": "int32", "BoolStorage": "bool",
       "DoubleStorage": "float64", "ByteStorage": "uint8", "BFloat16Storage": "uint16"}


class _Storage:
    def __init__(self, dtype, key):
        self.dtype, self.key = dtype, key


class _Tensor:
    def __init__(self, st, off, size, stride):
        self.st, self.off, self.size, self.stride = st, off, tuple(size), tuple(stride)


class _Unpickler(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda st, off, size, stride, *a: _Tensor(st, off, size, stride)
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda t, *a: t
        if mod == "torch" and name in _DT:
            return name
        return lambda *a, **k: f"<{mod}.{name}>"  # inert placeholder: nothing from the file is ever called

    def persistent_load(self, pid):
        _, cls, key, _, _ = pid
        return _Storage(_DT.get(cls, "float32"), key)


def state_dict(path, prefix="", arrays=True):
    """{name: np.ndarray} (or {name: (shape, dtype)} with arrays=False) for the tensors whose name starts with `prefix`."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    obj = _Unpickler(z.open(f"{root}/data.pkl")).load()
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    out = {}
    for k, t in sd.items():
        if not isinstance(t, _Tensor) or not k.startswith(prefix):
            continue
        if not arrays:
            out[k] = (t.size, t.st.dtype)
            continue
        buf = np.frombuffer(z.read(f"{root}/data/{t.st.key}"), dtype=t.st.dtype)
        out[k] = np.lib.stride_tricks.as_strided(buf[t.off:], shape=t.size, strides=[s * buf.itemsize for s in t.stride]).copy() if t.size else buf[t.off:t.off + 1].copy()
    return out
