#!/usr/bin/env bash
# ==============================================================================
# FlowEdit Enterprise TTS Service — IIT Server Launcher
# ==============================================================================

PORT="${1:-8000}"
HOST="${2:-0.0.0.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"
export FLOWEDIT_HOST="${HOST}"
export FLOWEDIT_PORT="${PORT}"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BOLD}${CYAN}   Starting FlowEdit Executive TTS Service on IIT Server${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "  Host: ${HOST}"
echo -e "  Port: ${PORT}"
echo -e "  Root: ${PROJECT_ROOT}"

# Detect Server IP
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${SERVER_IP}" ]; then
  SERVER_IP="<SERVER_IP>"
fi

echo -e "\n${GREEN}[✓] Web UI will be accessible at:${NC}"
echo -e "    ${BOLD}http://${SERVER_IP}:${PORT}/ui${NC}"
echo -e "    ${BOLD}http://localhost:${PORT}/ui${NC} (via SSH tunnel)\n"

echo -e "${YELLOW}Starting Uvicorn Server...${NC}\n"
cd "${PROJECT_ROOT}"
python3 -m uvicorn flowedit.api.main:app --host "${HOST}" --port "${PORT}" --workers 1
