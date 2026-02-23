#!/usr/bin/env bash
#
# Alfred AI — One-Line Installer
# Usage: curl -sL https://raw.githubusercontent.com/Aaron-A/Alfred-Ai/main/install.sh | bash
#
set -e

REPO="https://github.com/Aaron-A/Alfred-Ai.git"
INSTALL_DIR="${ALFRED_HOME:-$HOME/.alfred-ai}"
SYMLINK="/usr/local/bin/alfred"

echo ""
echo "  ╭──────────────────────────────────────╮"
echo "  │         Alfred AI — Installer         │"
echo "  ╰──────────────────────────────────────╯"
echo ""

# ─── Check OS ───────────────────────────────────────
OS="$(uname -s)"
if [[ "$OS" != "Darwin" && "$OS" != "Linux" ]]; then
    echo "  ✗ Unsupported OS: $OS (macOS or Linux required)"
    exit 1
fi

# ─── Check Python ──────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "  ✗ Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    echo "  ✗ Python $PY_VERSION found, but 3.10+ required."
    exit 1
fi
echo "  ✓ Python $PY_VERSION"

# ─── Check/Install uv ──────────────────────────────
if command -v uv &>/dev/null; then
    echo "  ✓ uv $(uv --version 2>/dev/null | head -1)"
else
    echo "  ⟳ Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
    export PATH="$HOME/.local/bin:$PATH"
    if command -v uv &>/dev/null; then
        echo "  ✓ uv installed"
    else
        echo "  ✗ Failed to install uv. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi

# ─── Check for existing install ─────────────────────
if [[ -d "$INSTALL_DIR" ]]; then
    echo ""
    echo "  ⚠  Existing installation found at $INSTALL_DIR"
    read -p "  Reinstall? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "  Cancelled."
        exit 0
    fi
    rm -rf "$INSTALL_DIR"
fi

# ─── Clone ──────────────────────────────────────────
echo ""
echo "  ⟳ Cloning Alfred AI..."
git clone --quiet "$REPO" "$INSTALL_DIR"
echo "  ✓ Cloned to $INSTALL_DIR"

# ─── Install dependencies ──────────────────────────
echo ""
echo "  ⟳ Installing dependencies..."
cd "$INSTALL_DIR"
uv sync --all-extras --quiet
echo "  ✓ Dependencies installed"

# ─── Symlink ────────────────────────────────────────
echo ""
if [[ -L "$SYMLINK" || -f "$SYMLINK" ]]; then
    echo "  ⚠  $SYMLINK already exists, updating..."
    sudo rm -f "$SYMLINK"
fi

if sudo ln -sf "$INSTALL_DIR/alfred" "$SYMLINK" 2>/dev/null; then
    echo "  ✓ Linked: alfred → $SYMLINK"
else
    echo "  ⚠  Could not create symlink (no sudo). Add to PATH manually:"
    echo "     export PATH=\"$INSTALL_DIR:\$PATH\""
fi

# ─── Done ───────────────────────────────────────────
echo ""
echo "  ╭──────────────────────────────────────╮"
echo "  │           Install complete!           │"
echo "  ╰──────────────────────────────────────╯"
echo ""
echo "  Next steps:"
echo "    alfred setup              # Configure LLM + create first agent"
echo "    alfred start              # Start Alfred"
echo ""
echo "  Installed to: $INSTALL_DIR"
echo ""
