#!/usr/bin/env bash
# generate_patches.sh
# Creates a patch file for each submodule that has local changes relative to
# its upstream tracking branch (or HEAD if no tracking branch exists).
# Patch files are written to the same directory as this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" || {
  echo "Error: not inside a git repository." >&2
  exit 1
}

OUTPUT_DIR="$SCRIPT_DIR"

patch_count=0
skipped_count=0

echo "Scanning submodules for local changes..."
echo "Output directory: $OUTPUT_DIR"
echo "-----------------------------------"

while IFS= read -r submodule_path; do
  [[ -z "$submodule_path" ]] && continue

  abs_path="$ROOT_DIR/$submodule_path"

  if [[ ! -d "$abs_path/.git" && ! -f "$abs_path/.git" ]]; then
    echo "[SKIP] $submodule_path — not initialized"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  pushd "$abs_path" > /dev/null

  # Staged + unstaged changes on tracked files
  dirty=$(git status --porcelain 2>/dev/null)

  # Commits not yet pushed to the tracked remote branch
  unpushed=""
  tracking=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
  if [[ -n "$tracking" ]]; then
    unpushed=$(git log "@{u}..HEAD" --oneline 2>/dev/null || true)
  fi

  if [[ -z "$dirty" && -z "$unpushed" ]]; then
    echo "[CLEAN] $submodule_path"
    popd > /dev/null
    continue
  fi

  # Replace path separators so the name is usable as a filename
  safe_name="${submodule_path//\//__}"
  patch_file="$OUTPUT_DIR/${safe_name}.patch"

  {
    echo "# Submodule: $submodule_path"
    echo "# Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo ""

    if [[ -n "$dirty" ]]; then
      echo "## Uncommitted changes ##"
      git diff HEAD \
        --src-prefix="a/$submodule_path/" \
        --dst-prefix="b/$submodule_path/" \
        2>/dev/null || true
      git diff --cached \
        --src-prefix="a/$submodule_path/" \
        --dst-prefix="b/$submodule_path/" \
        2>/dev/null || true
      echo ""
    fi

    if [[ -n "$unpushed" ]]; then
      echo "## Unpushed commits (vs $tracking) ##"
      git format-patch --stdout \
        --src-prefix="a/$submodule_path/" \
        --dst-prefix="b/$submodule_path/" \
        "@{u}..HEAD" 2>/dev/null || true
    fi
  } > "$patch_file"

  if [[ ! -s "$patch_file" ]]; then
    rm -f "$patch_file"
    echo "[SKIP] $submodule_path — no diff output (binary-only or untracked files only?)"
    skipped_count=$((skipped_count + 1))
  else
    echo "[PATCH] $submodule_path → $(basename "$patch_file")"
    patch_count=$((patch_count + 1))
  fi

  popd > /dev/null

done < <(git -C "$ROOT_DIR" submodule foreach --quiet --recursive 'echo "$displaypath"' 2>/dev/null)

echo "-----------------------------------"
echo "Done. Patches created: $patch_count | Skipped: $skipped_count"
echo "Patch files saved to: $OUTPUT_DIR"
