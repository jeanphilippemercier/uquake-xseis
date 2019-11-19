import numpy as np
import io


def array_to_bytes(dat):
    output = io.BytesIO()
    np.save(output, dat)
    return output.getvalue()


def bytes_to_array(buf):
    return np.load(io.BytesIO(buf))


def testf():
    print("hello")
