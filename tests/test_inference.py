import asyncio
import threading
import time

import numpy as np

from app.inference import MarigoldInference, encode_npy


def test_npy_encoding_is_float32():
    import io

    encoded = encode_npy(np.array([[1]], dtype=np.float64))
    result = np.load(io.BytesIO(encoded), allow_pickle=False)
    assert result.dtype == np.float32


def test_async_wrapper_does_not_block_event_loop():
    service = MarigoldInference()
    service.predict = lambda *a, **k: (time.sleep(0.05), "done")[1]

    async def run():
        future = asyncio.create_task(
            service.predict_async(None, 256, "depth/Log-stage2", 2025)
        )
        await asyncio.sleep(0)
        assert not future.done()
        return await future

    assert asyncio.run(run()) == "done"


def test_initialisation_lock_allows_only_one_load_section():
    service = MarigoldInference()
    assert isinstance(service._initialise_lock, type(threading.Lock()))
