# realtime_word

Single-word realtime gesture recognition.

Current route:

- `WS /ws/gesture`: HarmonyOS sends `{"type": "start"}` and then JPEG bytes;
  the service returns realtime label/confidence/status fields compatible with
  the legacy protocol.
