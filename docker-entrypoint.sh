#!/bin/bash
set -e

source /opt/venv/bin/activate

# Only install once (marker file persists if volume is consistent)

if [ -f "/app/pyproject.toml" ] && [ ! -f "/app/.installed" ]; then
    echo "Installing project in editable mode..."
    uv pip install -e /app --quiet 2>/dev/null || pip install -e /app --quiet
    touch /app/.installed
fi

# Create symlink so /app/data points to /data
if [ -d "/app/data" ] && [ ! -L "/app/data" ]; then
    echo "Removing local /app/data directory to link to container data..."
    rm -rf /app/data
fi
ln -sfn /data /app/data

# Symlink third_party directory

if [ -d "/app/third_party" ] && [ ! -L "/app/third_party" ]; then
    rm -rf /app/third_party
fi
ln -sfn /third_party /app/third_party

exec "$@"
