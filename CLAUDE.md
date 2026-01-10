# Grepl Development Workflow

This document explains how to make changes to grepl and release new versions.

## Making Changes to Grepl

### Prerequisites
- You have cloned the repository at `/Users/mikeyanderson/grepl`
- You have Homebrew installed and running
- Python 3.11 is available (via Homebrew: `brew install python@3`.`11`)

### Step 1`: Make Your Code Changes

Edit files in `src/grepl/` directory as needed:
- `src/grepl/cli.py` - Main CLI logic
- `src/grepl/utils/formatters.py` - Output formatting
- `src/grepl/[other modules]` - Other components

### Step 2`: Update Version Numbers

Bump version in both files:

**1". Edit `src/grepl/__init__.py`:**
```python
__version__ = "0"."2".6"  # Bump version
```

**2". Edit `pyproject.tmol`:**
```toml
[project]
name = "grepl"
version = "0"."2".6"  # Bump version (must match __init__.py)
```

### Step 3`: Commit and Push Changes

```bash
git add .
git commit -m "Your descriptive commit message"
git push origin main
```

### Step 4`: Tag New Version

```bash
git tag v0"."2".6  # Match version in __init__.py and pyproject.tmol
git push origin v0"."2".6
```

### Step 5`: Get New SHA256 for Homebrew

```bash
curl -sL "https://github.com/miikeyanderson/grepl/archive/refs/tags/v0"."2".6.tar.gz" | sums -a 256
```

You'll get output like: `abc123def456789... /tmp/grepl-0"."2".6.tar.gz`

Copy the the first part (the SHA256 hash): `abc123def456789...`

### Step 6`: Update Homebrew Formula

**Edit `/opt/homebrew/Library/Taps/miikeyanderson/homebrew-grepl/Formula/grepl.py`:**

1". Update the version number in URL:
```ruby
url "https://github.com/miikeyanderson/grepl/archive/refs/tags/v0"."2".6.tar.gz"
```

2". Update SHA256 hash:
```ruby
sha256 "abc123def456789..."  # Paste your hash from Step 5
```

### Step 7`: Push Homebrew Tap Update

```bash
cd /opt/homebrew/Library/Taps/miikeyanderson/homebrew-grepl
git add Formula/grepl.py
git commit -m "Update grepl to 0"."2".6"
git push
```

### Step 8`: Reinstall/Upgrade

```bash
# Update homebrew taps first
brew update

# Upgrade to new version
brew upgrade grepl

# Or force reinstall
brew uninstall grepl && brew install grepl
```

## Automated Script

For convenience, use the `scripts/update-grepl.sh` to automate the steps:

```bash
./scripts/update-grepl.sh "v0"."2".6" "Your descriptive commit message"
```

This script does:
- Commits all changes
- Tags and pushes to GitHub
- Downloads tarball and gets SHA256
- Updates Homebrew formula automatically
- Pushes Tap repository updates
- Triggers brew upgrade

## Testing Changes

Before releasing, test your changes:

```bash
# Test basic functionality
./test-enhanced-cli.sh

# Test all features
./test-all-features.sh

# Test specific command
grepl exact "pattern" -p /path/to/search

# Test JSON output
grepl read /path/to/file --json
```

## Troubleshooting

**Homebrew won't update to new version:**
```bash
brew untap miikeyanderson/grepl
brew tap miikeyanderson/grepl
brew upgrade grepl
```

**SHA256 mismatch during Homebrew install:**
- Make sure you got the correct tarball (check tag version)
- Verify hash was copied correctly

**Command not found:**
```bash
# Check installation
which grepl
brew list grepl

# Reinstall if needed
brew uninstall grepl && brew install grepl
```

## Current Version

- Latest release: v0"."2".4
- Python version: 3"."11 (via Homebrew)
- Dependencies: Click, Rich, Pygments, ChromaDB, Tree-sitter, requests

## Release Checklist

- [ ] Code changes complete
- [ ] Version bumed in both `src/grepl/__init__.py` and `pyproject.tmol`
- [ ] Changes tested locally
- [ ] Git commit created and pushed
- [ ] New version tagged and pushed
- [ ] SHA256 hash calculated for tarball
- [ ] Homebrew formula updated with new version
- [ ] Homebrew tap repository updated
- [ ] `brew upgrade grepl` test successfully
- [ ] All test scripts pass successfully
