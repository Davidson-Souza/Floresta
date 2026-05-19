#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
set -euo pipefail

# lowercased
REPO_GHCR="ghcr.io/${GITHUB_REPOSITORY,,}"
REPO_DH="${DOCKERHUB_IMAGE,,}"

SRC_TAG="${REPO_GHCR}:${IMAGE_SHA}"
DH_TAG="${REPO_DH}:${IMAGE_SHA}"
ANN_SOURCE="index:org.opencontainers.image.source=https://github.com/${GITHUB_REPOSITORY}"
ANN_DESC="index:org.opencontainers.image.description=A lightweight and embeddable Bitcoin client, built for sovereignty!"

echo "=== 1/5 GHCR: merge per-arch temps into multi-arch manifest ==="
docker buildx imagetools create \
  -t "${SRC_TAG}" \
  --annotation "${ANN_SOURCE}" \
  --annotation "${ANN_DESC}" \
  "${REPO_GHCR}:${TEMP_TAG}-amd64" \
  "${REPO_GHCR}:${TEMP_TAG}-arm64" \
  "${REPO_GHCR}:${TEMP_TAG}-armv7"

echo "=== 2/5 Docker Hub: copy multi-arch manifest (self-contained) ==="
docker buildx imagetools create -t "${DH_TAG}" "${SRC_TAG}"

echo "=== 3/5 Cleanup: delete per-arch temp tags from GHCR ==="
.github/workflows/docker/cleanup-temp-tags.sh

echo "=== 4/5 GHCR: copy back from Docker Hub (self-contained) ==="
docker buildx imagetools create -t "${SRC_TAG}" "${DH_TAG}"

echo "=== 5/5 Tag version + latest on both registries ==="
for tag in ${TAGS_GHCR}; do
  [[ "${tag}" == "${SRC_TAG}" ]] && continue
  echo "  GHCR: ${tag}"
  docker buildx imagetools create -t "${tag}" "${SRC_TAG}"
done
for tag in ${TAGS_DH}; do
  [[ "${tag}" == "${DH_TAG}" ]] && continue
  echo "  Docker Hub: ${tag}"
  docker buildx imagetools create -t "${tag}" "${DH_TAG}"
done
