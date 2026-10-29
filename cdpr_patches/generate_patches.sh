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
removed_count=0

# Deletes the patch left over from a previous run when the submodule no longer
# has local changes, so it is not applied by the build scripts.
remove_stale_patch() {
  if [[ -f "$patch_file" ]]; then
    rm -f "$patch_file"
    echo "[REMOVED] $(basename "$patch_file") — submodule has no local changes"
    removed_count=$((removed_count + 1))
  fi
}

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

  # Replace path separators so the name is usable as a filename
  safe_name="${submodule_path//\//__}"
  patch_file="$OUTPUT_DIR/${safe_name}.patch"

  pushd "$abs_path" > /dev/null

  # Staged, unstaged, and untracked (non-ignored) changes
  dirty=$(git status --porcelain 2>/dev/null)

  # Commits not yet pushed to the tracked remote branch
  unpushed=""
  tracking=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)
  if [[ -n "$tracking" ]]; then
    unpushed=$(git log "@{u}..HEAD" --oneline 2>/dev/null || true)
  fi

  if [[ -z "$dirty" && -z "$unpushed" ]]; then
    echo "[CLEAN] $submodule_path"
    remove_stale_patch
    popd > /dev/null
    continue
  fi

  # Diff the working tree (including untracked files) against HEAD. Untracked
  # files are added to a temporary copy of the index so the real index is
  # left untouched.
  uncommitted_diff=""
  if [[ -n "$dirty" ]]; then
    tmp_index=$(mktemp)
    index_file=$(git rev-parse --git-path index)
    [[ -f "$index_file" ]] && cp "$index_file" "$tmp_index"
    GIT_INDEX_FILE="$tmp_index" git add -A
    uncommitted_diff=$(GIT_INDEX_FILE="$tmp_index" git diff --cached --binary HEAD \
      --src-prefix="a/$submodule_path/" \
      --dst-prefix="b/$submodule_path/" \
      2>/dev/null || true)
    rm -f "$tmp_index"
  fi

  unpushed_patch=""
  if [[ -n "$unpushed" ]]; then
    unpushed_patch=$(git format-patch --stdout --binary \
      --src-prefix="a/$submodule_path/" \
      --dst-prefix="b/$submodule_path/" \
      "@{u}..HEAD" 2>/dev/null || true)
  fi

  if [[ -z "$uncommitted_diff" && -z "$unpushed_patch" ]]; then
    echo "[SKIP] $submodule_path — no diff output"
    skipped_count=$((skipped_count + 1))
    remove_stale_patch
    popd > /dev/null
    continue
  fi

  # Unpushed commits come first: the uncommitted diff is relative to HEAD, so
  # it must be applied on top of them.
  body=$(
    if [[ -n "$unpushed_patch" ]]; then
      echo "## Unpushed commits (vs $tracking) ##"
      printf '%s\n' "$unpushed_patch"
    fi
    if [[ -n "$uncommitted_diff" ]]; then
      echo "## Uncommitted changes ##"
      printf '%s\n' "$uncommitted_diff"
    fi
  )

  # Leave an existing patch alone if only the timestamp would change.
  if [[ -f "$patch_file" ]] && [[ "$(tail -n +4 "$patch_file")" == "$body" ]]; then
    echo "[UNCHANGED] $submodule_path → $(basename "$patch_file")"
  else
    {
      echo "# Submodule: $submodule_path"
      echo "# Generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
      echo ""
      printf '%s\n' "$body"
    } > "$patch_file"
    echo "[PATCH] $submodule_path → $(basename "$patch_file")"
  fi
  patch_count=$((patch_count + 1))

  popd > /dev/null

done < <(git -C "$ROOT_DIR" submodule foreach --quiet --recursive 'echo "$displaypath"' 2>/dev/null)

echo "-----------------------------------"
echo "Done. Patches created: $patch_count | Skipped: $skipped_count | Removed: $removed_count"
echo "Patch files saved to: $OUTPUT_DIR"
