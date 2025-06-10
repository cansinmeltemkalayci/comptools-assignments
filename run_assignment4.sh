#!/bin/bash
# Script to run Assignment4 code and save output to file
cd assignment4

# Create output directory if it doesn't exist
mkdir -p output

# Run the python script and save output to a text file
python assignment4_python.py > output/assignment4_results.txt 2>&1

# Copy the generated images to output directory if they exist
if [ -f "coefficient_recovery.png" ]; then
  cp coefficient_recovery.png output/
fi

if [ -f "cross_validation.png" ]; then
  cp cross_validation.png output/
fi

echo "Assignment 4 completed. Results saved to assignment4/output/"
cd ..
