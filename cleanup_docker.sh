#!/bin/bash
# Docker cleanup script - removes all unused images, containers, volumes

set -e

echo "🧹 Docker Cleanup Script"
echo "========================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Current disk usage:${NC}"
df -h / | grep -v Filesystem

echo ""
echo -e "${BLUE}Docker disk usage before cleanup:${NC}"
docker system df

echo ""
echo -e "${YELLOW}⚠️  This will remove:${NC}"
echo "  - All stopped containers"
echo "  - All unused images"
echo "  - All unused volumes"
echo "  - All unused networks"
echo "  - Build cache"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo -e "${BLUE}Step 1: Stopping all containers...${NC}"
docker compose down 2>/dev/null || true
docker stop $(docker ps -aq) 2>/dev/null || echo "No running containers"

echo ""
echo -e "${BLUE}Step 2: Removing stopped containers...${NC}"
docker container prune -f

echo ""
echo -e "${BLUE}Step 3: Removing unused images...${NC}"
docker image prune -a -f

echo ""
echo -e "${BLUE}Step 4: Removing unused volumes...${NC}"
docker volume prune -f

echo ""
echo -e "${BLUE}Step 5: Removing unused networks...${NC}"
docker network prune -f

echo ""
echo -e "${BLUE}Step 6: Removing build cache...${NC}"
docker builder prune -a -f

echo ""
echo -e "${GREEN}✓ Cleanup complete!${NC}"
echo ""
echo -e "${BLUE}Docker disk usage after cleanup:${NC}"
docker system df

echo ""
echo -e "${BLUE}System disk usage after cleanup:${NC}"
df -h / | grep -v Filesystem

echo ""
echo -e "${GREEN}Done! You can now rebuild your images.${NC}"
