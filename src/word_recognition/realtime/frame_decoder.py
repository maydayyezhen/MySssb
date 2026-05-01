# -*- coding: utf-8 -*-
"""Realtime recognition frame decoder."""

import cv2
import numpy as np


def decode_frontend_frame(bytes_data: bytes):
    """
    Decode frontend-supplied MediaPipe-ready JPEG bytes.

    No rotate, mirror, crop, or resize is applied here.
    """
    arr = np.frombuffer(bytes_data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
