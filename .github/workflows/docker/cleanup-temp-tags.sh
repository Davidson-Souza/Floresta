#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
set -euo pipefail

if [[ "${CLEANUP_TEMP_TAGS:-true}" == "false" || "${CLEANUP_TEMP_TAGS:-true}" == "0" ]]; then
  echo "CLEANUP_TEMP_TAGS=${CLEANUP_TEMP_TAGS}, skipping cleanup"
  exit 0
fi

# lowercased
OWNER="${GITHUB_REPOSITORY_OWNER,,}"
PKG="${GITHUB_REPOSITORY,,}";
# extract repository name
PKG="${PKG##*/}"

if gh api "/orgs/${OWNER}" --silent 2>/dev/null; then
  BASE="/orgs/${OWNER}/packages/container/${PKG}"
else
  BASE="/users/${OWNER}/packages/container/${PKG}"
fi

PREFIX="${TEMP_TAG}"
for ARCH in amd64 arm64 armv7; do
  TAG="${PREFIX}-${ARCH}"
  VID=$(gh api "${BASE}/versions" \
    --jq ".[] | select(.metadata.container.tags | index(\"${TAG}\")) | .id" \
    2>/dev/null || true)
  if [[ -n "${VID}" ]]; then
    gh api --method DELETE "${BASE}/versions/${VID}" --silent 2>/dev/null || true
    echo "Deleted ${TAG} (version ${VID})"
  else
    echo "Tag ${TAG} not found, skipping"
  fi
done
