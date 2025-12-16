#!/bin/bash
# Rewrite commit message to remove Section 9 reference

cd /c/Scratch/bloodvital-ml

# Use filter-branch to rewrite commit message
git filter-branch -f --msg-filter '
if [ "$GIT_COMMIT" = "3afef75" ] || echo "$GIT_COMMIT" | grep -q "3afef75"; then
  echo "docs: Enhance Conclusion with broader impact statement

Adds generalizability discussion to strengthen paper conclusion.
Refines academic narrative for journal submission."
else
  cat
fi
' -- --all
