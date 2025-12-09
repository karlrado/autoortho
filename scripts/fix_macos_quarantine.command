#!/bin/bash
#
# AutoOrtho - Fix macOS Quarantine
#
# This script removes the quarantine attribute from AutoOrtho.app
# Run this if you see "Error -47" or the app won't open after download.
#
# Double-click this file to run it, or run from Terminal:
#   ./fix_macos_quarantine.command
#

echo "================================================"
echo "  AutoOrtho - Fix macOS Quarantine Attribute"
echo "================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP_PATH="$SCRIPT_DIR/AutoOrtho.app"

# Check if AutoOrtho.app exists in the same directory
if [ ! -d "$APP_PATH" ]; then
    echo "ERROR: AutoOrtho.app not found in the same folder as this script."
    echo ""
    echo "Please make sure this script is in the same folder as AutoOrtho.app"
    echo "Expected location: $APP_PATH"
    echo ""
    read -p "Press Enter to exit..."
    exit 1
fi

echo "Found AutoOrtho.app at: $APP_PATH"
echo ""
echo "Removing quarantine attribute..."
echo ""

# Remove the quarantine attribute recursively
xattr -cr "$APP_PATH"

if [ $? -eq 0 ]; then
    echo "SUCCESS! Quarantine attribute removed."
    echo ""
    echo "You can now open AutoOrtho.app normally."
    echo ""
else
    echo "ERROR: Failed to remove quarantine attribute."
    echo ""
    echo "Try running this command manually in Terminal:"
    echo "  xattr -cr \"$APP_PATH\""
    echo ""
fi

read -p "Press Enter to exit..."

