#!/bin/bash

set -x

SEARCH_STRING="$1"

# Check if the search string is provided
if [[ -z $SEARCH_STRING ]]; then
  echo "Please provide a search string."
  exit 1
fi

PIDS=$(pgrep -f "$SEARCH_STRING")

if [[ -z $PIDS ]]; then
  echo "No processes found matching the search string."
  exit 0
fi

echo "Processes found with the search string '$SEARCH_STRING':"
echo "$PIDS"

read -p "Are you sure you want to kill these processes? (y/n): " CONFIRMATION

if [[ $CONFIRMATION != "y" ]]; then
  echo "Aborted."
  exit 0
fi

# Kill the processes
for PID in $PIDS; do
  echo "Killing process with PID $PID..."
  kill $PID
done

echo "All processes killed successfully."