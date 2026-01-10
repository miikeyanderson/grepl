#!/bin/bash

# Grepl Update Script - Automates the version release and Homebrew update

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

set -e  # Exit on error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <version> <commit-message>"
    echo "Example: $0 v0.2".6" \"Add new feature\""
    exit 1
fi

VERSION=$1
COMMIT_MESSAGE=$2
GREPL_DIR="/Users/mikeyanderson/grepl"
HOMEBREW_TAP_DIR="/opt/homebrew/Library/Taps/miikeyanderson/homebrew-grepl"
FORMULA_PATH="$HOMEBREW_TAP_DIR/Formula/grepl.rb"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Grepl Update Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Validate version format
if [[ ! $VERSION =~ ^v0\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Error: Version must be in format v0.X.Y${NC}"
    exit 1
fi

# Extract version number (without 'v' prefix)
VERSION_NUM=${VERSION#v}

echo -e "${GREEN}Step 1: Commit changes${NC}"
echo "----------------------------"
cd "$GREPL_DIR"

# Check if there are changes to commit
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}No changes to commit. Commit the skipped.${NC}"
else
    git add .
    git commit -m "$COMMIT_MESSAGE"
    echo -e "${GREEN}Changes committed!${NC}"
fi
echo ""

echo -e "${GREEN}Step 2: Push to GitHub${NC}"
echo "----------------------------"
git push origin main
echo -e "${GREEN}Pushed to main!${NC}"
echo ""

echo -e "${GREEN}Step 3: Tag version${NC}"
echo "----------------------------"
git tag "$VERSION"
git push origin "$VERSION"
echo -e "${GREEN}Tagged and pushed $VERSION!${NC}"
echo ""

echo -e "${GREEN}Step 4: Get SHA256 hash${NC}"
echo "----------------------------"
TARBALL_URL="https://github.com/miikeyanderson/grepl/archive/refs/tags/${VERSION}.tar.gz"
echo "Downloading ${TARBALL_URL}..."

# Use shasum (available on macOS by default)
SHA256=$(curl -sL "$TARBALL_URL" | shasum -a 256 | awk '{print $1}')

if [ -z "$SHA256" ]; then
    echo -e "${RED}Error: Failed to get SHA256 hash${NC}"
    exit 1
fi

echo -e "${GREEN}SHA256: $SHA256${NC}"
echo ""

echo -e "${GREEN}Step 5: Update Homebrew formula${NC}"
echo "----------------------------"

# Backup old formula
cp "$FORMULA_PATH" "${FORMULA_PATH}.backup"

# Update the formula
python3 << EOF
import re

formula_path = "$FORMULA_PATH"
tag = "$VERSION"
sha256 = "$SHA256"

with open(formula_path, 'r') as f:
    content = f.read()

# Update URL
content = re.sub(
    r'url "https://github.com/miikeyanderson/grepl/archive/refs/tags/v[0-9]+\.[0-9]+\.[0-9]+\.tar.gz"',
    f'url "https://github.com/miikeyanderson/grepl/archive/refs/tags/{tag}.tar.gz"',
    content
)

# Update SHA256
content = re.sub(
    r'sha256 "[a-f0-9]+"',
    f'sha256 "{sha256}"',
    content
)

with open(formula_path, 'w') as f:
    f.write(content)

print(f"Updated formula to {tag}")
EOF

echo -e "${GREEN}Homebrew formula updated!${NC}"
echo ""

echo -e "${GREEN}Step 6: Push Homebrew tap${NC}"
echo "----------------------------"
cd "$HOMEBREW_TAP_DIR"
rm -f "${FORMULA_PATH}.backup" || true

if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}No Homebrew formula changes to commit. Skipping tap update.${NC}"
else
    git add Formula/grepl.rb
    git commit -m "Update grepl to $VERSION_NUM"
    git push
    echo -e "${GREEN}Homebrew tap updated!${NC}"
fi
echo ""

echo -e "${GREEN}Step 7: Upgrade Homebrew package${NC}"
echo "----------------------------"
cd ~
brew update
brew upgrade grepl
echo -e "${GREEN}Grepl upgraded to $VERSION!${NC}"
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Update Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Testing installation:"
echo "  grepl --help"
echo "  grepl --version"
echo ""
echo -e "${YELLOW}To rollback if needed:${NC}"
echo "  git checkout previous_version"
echo "  brew uninstall grepl && brew install grepl"
