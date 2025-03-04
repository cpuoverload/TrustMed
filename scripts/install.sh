#!/bin/bash

set -e

# Install Poetry
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found, installing..."
    curl -sSL https://install.python-poetry.org | python3 -
else
    echo "Poetry already installed, skipping installation."
fi
