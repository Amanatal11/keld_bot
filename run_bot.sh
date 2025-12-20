#!/bin/bash
# Run the bot using the virtual environment python
if [ -f ".venv/bin/python3" ]; then
    .venv/bin/python3 bot.py
else
    echo "Virtual environment not found in .venv/"
    echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi
