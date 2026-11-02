#!/usr/bin/env bash
# True one-command installer:
#   curl -fsSL https://raw.githubusercontent.com/pranava-mk/Switch_toggle_using_openCV/main/install.sh | bash
#
# Works with nothing pre-cloned: checks deps (offers to install them via the
# detected OS package manager, asking first), clones the repo if needed,
# creates a venv, installs the pinned requirements, and prints the run
# commands. Does NOT auto-launch any detection script (it needs a real
# webcam, which may not exist in the install environment).
set -euo pipefail

REPO_URL="https://github.com/pranava-mk/Switch_toggle_using_openCV.git"
REPO_DIR="Switch_toggle_using_openCV"

log()  { printf '==> %s\n' "$1"; }
warn() { printf '!! %s\n' "$1" >&2; }

confirm() {
    # confirm "question" -> 0 if yes. Falls back to "no" if there's no
    # interactive tty available (e.g. piped via curl|bash with stdin consumed,
    # or a sandboxed/headless exec environment with no controlling terminal).
    local reply="n"
    if read -r -p "$1 [y/N] " reply 2>/dev/null </dev/tty; then
        :
    elif [[ -t 0 ]] && read -r -p "$1 [y/N] " reply; then
        :
    else
        warn "No interactive terminal available to prompt; defaulting to 'no'."
        reply="n"
    fi
    [[ "$reply" =~ ^[Yy]$ ]]
}

detect_pkg_mgr() {
    if command -v apt-get >/dev/null 2>&1; then echo apt
    elif command -v pacman >/dev/null 2>&1; then echo pacman
    elif command -v brew >/dev/null 2>&1; then echo brew
    else echo none
    fi
}

install_pkg() {
    local pkg_apt="$1" pkg_pacman="$2" pkg_brew="$3"
    local mgr; mgr="$(detect_pkg_mgr)"
    case "$mgr" in
        apt)
            if confirm "Install '$pkg_apt' via apt-get (sudo required)?"; then
                sudo apt-get update && sudo apt-get install -y "$pkg_apt"
            else
                warn "Skipped installing $pkg_apt; continuing may fail."
            fi
            ;;
        pacman)
            if confirm "Install '$pkg_pacman' via pacman (sudo required)?"; then
                sudo pacman -Sy --noconfirm "$pkg_pacman"
            else
                warn "Skipped installing $pkg_pacman; continuing may fail."
            fi
            ;;
        brew)
            if confirm "Install '$pkg_brew' via brew?"; then
                brew install "$pkg_brew"
            else
                warn "Skipped installing $pkg_brew; continuing may fail."
            fi
            ;;
        none)
            warn "No known package manager (apt/pacman/brew) detected."
            warn "Please install '$pkg_apt' manually and re-run this script."
            exit 1
            ;;
    esac
}

# ---- 1. Check prerequisites -------------------------------------------------
log "Checking prerequisites (git, python3, pip)..."

if ! command -v git >/dev/null 2>&1; then
    warn "git not found."
    install_pkg git git git
fi

if ! command -v python3 >/dev/null 2>&1; then
    warn "python3 not found."
    install_pkg python3 python python3
fi

if ! python3 -m pip --version >/dev/null 2>&1; then
    warn "pip (python3 -m pip) not found."
    install_pkg python3-pip python-pip python3
fi

if ! python3 -c "import venv" >/dev/null 2>&1; then
    warn "python3 'venv' module not found."
    install_pkg python3-venv python python3
fi

for tool in git python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        warn "$tool is still missing after the install attempt. Aborting."
        exit 1
    fi
done

# ---- 2. Clone the repo (if not already inside/present) ---------------------
if [[ -f "./requirements.txt" && -f "./setup.sh" && -f "./install.sh" ]]; then
    log "Already inside the repo checkout; using current directory."
else
    if [[ -d "$REPO_DIR" ]]; then
        log "Found existing '$REPO_DIR' directory; using it."
    else
        log "Cloning $REPO_URL ..."
        git clone "$REPO_URL" "$REPO_DIR"
    fi
    cd "$REPO_DIR"
fi

log "Working directory: $(pwd)"

# ---- 3. Create/use a virtualenv and install pinned requirements ------------
if [[ ! -d ".venv" ]]; then
    log "Creating virtualenv (.venv)..."
    python3 -m venv .venv
else
    log "Reusing existing virtualenv (.venv)."
fi

log "Installing dependencies from requirements.txt (opencv-python pinned to 4.10.0.84 for Caffe support)..."
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements.txt

# ---- 4. Done. Print run commands (do NOT auto-launch: needs a real webcam) --
echo
log "Install complete."
cat <<'EOF'

A real webcam is required to run detection — this installer will NOT
auto-launch anything. When you're ready, run one of:

  .venv/bin/python "human detection.py"              # detection only
  .venv/bin/python human_detection_switch_toggle.py   # detection + on-screen toggle
  .venv/bin/python Simulation.py                      # detection + PyQt5 status window

Press 'q' to quit any of them.
EOF
