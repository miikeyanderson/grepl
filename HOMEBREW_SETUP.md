# Homebrew Setup for Grepl

To install grepl via Homebrew without Python version issues, follow these steps:

## 1. Create GitHub Tap Repository

Create a new GitHub repository called `homebrew-grepl` at:
https://github.com/miikeyanderson/homebrew-grepl

## 2. Push Formula to Tap

```bash
cd /tmp/homebrew-grepl
git init
git add Formula/grepl.rb
git commit -m "Add grepl formula"
git branch -M main
git remote add origin https://github.com/miikeyanderson/homebrew-grepl.git
git push -u origin main
```

## 3. Install via Homebrew

```bash
brew tap miikeyanderson/grepl
brew install grepl
```

## 4. Verify Installation

```bash
grepl --help
```

## Formula Details

- **Name**: grepl
- **Version**: 0.2.2
- **Python Version**: 3.11 (uses Python@3.11 from Homebrew)
- **Dependencies**: ChromaDB, Ollama, Rich, Click, Pygments

## Why Homebrew?

Using Homebrew with Python@3.11 avoids the dependency conflicts you experienced with Python 3.14 (onnxruntime compatibility issues). Homebrew manages the Python environment separately and ensures all dependencies work together.

## Updating the Formula

When you release new versions:

1. Tag your commit: `git tag v0.2.3`
2. Push tag: `git push origin v0.2.3`
3. Download tarball and get SHA256:
```bash
curl -sL "https://github.com/miikeyanderson/grepl/archive/refs/tags/v0.2.3.tar.gz" | shasum -a 256
```
4. Update the formula:
```bash
# Edit version url and sha256 in homebrew-grepl/Formula/grepl.rb
git add Formula/grepl.rb
git commit -m "Update grepl to 0.2.3"
git push
```
5. Update: `brew upgrade grepl`

## Troubleshooting

If installation fails:
```bash
brew doctor
brew update
brew uninstall grepl
brew install grepl
```
