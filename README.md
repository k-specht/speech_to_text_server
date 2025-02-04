# Speech to Text Server
A simple STT server.

## Installation
1. Install the venv Python module (if needed).
2. Run `/path/to/your/python -m venv ./.venv && python3 speech_to_text_server.py`.
3. (optional) Move the module initialization in the script from the global variable to the function to avoid constant RAM usage.
4. (optional) If using GPU, double check the CUDA line in the Python script.
