#!/bin/bash
set -e

# ==============================================================================
# Setup 2GB Swap Memory on Ubuntu EC2 Host
# ==============================================================================

SWAP_FILE="/swapfile"
SWAP_SIZE="2G"

echo "Checking current swap status..."
if swapon --show | grep -q "${SWAP_FILE}"; then
    echo "Swap file ${SWAP_FILE} is already active!"
    free -h
    exit 0
fi

echo "Creating ${SWAP_SIZE} swapfile at ${SWAP_FILE}..."
sudo fallocate -l ${SWAP_SIZE} ${SWAP_FILE} || sudo dd if=/dev/zero of=${SWAP_FILE} bs=1M count=2048 status=progress

echo "Setting secure permissions (chmod 600)..."
sudo chmod 600 ${SWAP_FILE}

echo "Formatting swap partition..."
sudo mkswap ${SWAP_FILE}

echo "Activating swap..."
sudo swapon ${SWAP_FILE}

echo "Configuring automatic mount in /etc/fstab..."
if ! grep -q "${SWAP_FILE}" /etc/fstab; then
    echo "${SWAP_FILE} none swap sw 0 0" | sudo tee -a /etc/fstab
fi

# Set optimal swappiness for an EC2 server running containers (10 means only swap when RAM is 90% full)
sudo sysctl vm.swappiness=10
if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness=10" | sudo tee -a /etc/sysctl.conf
fi

echo "Swap configuration successfully completed!"
free -h
