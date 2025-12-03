#!/bin/bash
echo "--- Updating Decision Engine Suite ---"
git add .
echo "Enter a description for this update (e.g. 'Fixed bug'):"
read msg
git commit -m "$msg"
echo "--- Pulling latest changes from GitHub... ---"
git pull --rebase origin main
echo "--- Pushing your changes... ---"
git push
echo "--- Done! Streamlit Cloud will update automatically in a few moments. ---"
