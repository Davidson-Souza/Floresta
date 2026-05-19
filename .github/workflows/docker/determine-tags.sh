#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
set -euo pipefail

# lowercased
REPO_GHCR="ghcr.io/${GITHUB_REPOSITORY,,}"
REPO_DH="${DOCKERHUB_IMAGE,,}"

TAGS_GHCR="${REPO_GHCR}:${IMAGE_SHA}"
TAGS_DH="${REPO_DH}:${IMAGE_SHA}"

if [[ "${GITHUB_EVENT_NAME}" == "workflow_dispatch" ]]; then
  TAGS_GHCR="${TAGS_GHCR} ${REPO_GHCR}:${DISPATCH_TAG} ${REPO_GHCR}:latest"
  TAGS_DH="${TAGS_DH} ${REPO_DH}:${DISPATCH_TAG} ${REPO_DH}:latest"
elif [[ "${GITHUB_REF_TYPE}" == "tag" ]]; then
  TAGS_GHCR="${TAGS_GHCR} ${REPO_GHCR}:${GITHUB_REF_NAME} ${REPO_GHCR}:latest"
  TAGS_DH="${TAGS_DH} ${REPO_DH}:${GITHUB_REF_NAME} ${REPO_DH}:latest"
else
  TAGS_GHCR="${TAGS_GHCR} ${REPO_GHCR}:master"
  TAGS_DH="${TAGS_DH} ${REPO_DH}:master"
fi

echo "tags_ghcr=${TAGS_GHCR}" >> "${GITHUB_OUTPUT}"
echo "tags_dh=${TAGS_DH}" >> "${GITHUB_OUTPUT}"
