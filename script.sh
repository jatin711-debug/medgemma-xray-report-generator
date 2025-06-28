#!/bin/bash

# Update package lists
echo "Updating package lists..."
sudo apt-get update

# Install vim
echo "Installing vim..."
sudo apt install -y vim

echo "creating a .env file..."
touch .env

# Install Python requirements
echo "Installing Python requirements..."
pip install -r ./requirements.txt

echo "Setup complete!"
