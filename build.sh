#!/bin/bash
echo "Starting FreshTrack backend build..."

# Clean install
echo "Cleaning and installing dependencies..."
rm -rf node_modules package-lock.json
npm cache clean --force
npm install --production

echo "Build completed successfully!"
