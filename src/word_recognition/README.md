# word_recognition

Single-word gesture recognition service module.

This module owns the complete service surface for word-level recognition:

- realtime WebSocket recognition
- raw phone dataset collection
- raw-to-feature conversion
- training orchestration and artifact download
- runtime model reload/status APIs

The public paths stay compatible with the original FastAPI service.
