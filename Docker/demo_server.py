import lunavox_tts as lunavox

# Start server
lunavox.start_server(
    host="0.0.0.0",  # Host address
    port=9999,  # Port
    workers=1  # Number of workers
)