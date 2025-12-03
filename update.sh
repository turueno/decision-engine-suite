#!/bin/bash
echo "--- Updating Decision Engine Suite ---"
git add .
echo "Enter a description for this update (e.g. 'Fixed bug'):"
read msg
git commit -m "$msg"
git push
echo "--- Done! Streamlit Cloud will update automatically in a few moments. ---"
