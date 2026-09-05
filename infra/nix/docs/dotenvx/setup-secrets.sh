#!/usr/bin/env bash
# infra/nix/docs/dotenvx/setup-secrets.sh — One-time onboarding script for dotenvx secrets.
#
# Usage:
#   ./infra/nix/docs/dotenvx/setup-secrets.sh
#
# This script:
#   1. Reads .env.example to discover required keys.
#   2. Prompts you for each value.
#   3. Writes a temporary .env file.
#   4. Encrypts the values in place in .env (dotenvx >=1 has no .env.vault).
#   5. Stores the decryption key in your OS keyring.
#   6. Deletes .env.keys from disk (the encrypted .env is the secret store).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Ask git for the root rather than counting `..` levels: this script has already
# moved once (to infra/nix/docs/dotenvx/), and the hard-coded hop silently
# resolved to infra/nix, where there is no .env.example.
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if ! command -v dotenvx &>/dev/null; then
  echo "❌ dotenvx not found. Run 'nix develop' or 'direnv allow' first."
  exit 1
fi

# Checked BEFORE any plaintext is written. secret-tool is an external
# prerequisite (GNOME Keyring / KDE Wallet), deliberately not in the dev shell.
# Without this guard the script prompts for secrets, encrypts them, and only
# then dies on the keyring step -- leaving plaintext .env and .env.keys behind.
if ! command -v secret-tool &>/dev/null; then
  echo "❌ secret-tool not found; the decryption key could not be stored."
  echo "   Install libsecret (GNOME Keyring or KDE Wallet) and ensure a keyring"
  echo "   daemon is running, or skip the vault and use a plain gitignored .env."
  exit 1
fi

if [ ! -f .env.example ]; then
  echo "❌ .env.example not found. Cannot determine required keys."
  exit 1
fi

# Scrub the on-disk key ONLY once it is verifiably in the keyring. Until then
# .env.keys is the sole copy of the key that decrypts .env, so an
# unconditional scrub-on-exit turns any interrupt -- say, a hang on a locked
# keyring -- into permanent loss of every secret in the file. Warn instead.
KEY_SECURED=0
scrub_plaintext() {
  if [ "$KEY_SECURED" = "1" ]; then
    rm -f "$REPO_ROOT/.env.keys"
  elif [ -f "$REPO_ROOT/.env.keys" ]; then
    echo ""
    echo "⚠️  Exited before the key reached the keyring."
    echo "   .env.keys is KEPT — it is the only key that decrypts .env."
    echo "   Do not delete it until 'secret-tool lookup service"
    echo "   pricing-perspective key DOTENV_PRIVATE_KEY' returns it."
  fi
}
trap scrub_plaintext EXIT INT TERM

echo "🔑 Pricing Perspective — Secret Provisioner"
echo "============================================"
echo ""
echo "This script will prompt you for secret values, encrypt them into"
echo "in place in .env, and store the decryption key in your OS keyring."
echo ""

# Parse keys from .env.example (non-comment, non-empty lines)
KEYS=()
while IFS= read -r line; do
  # Skip comments and blank lines
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "$line" ]] && continue
  # Extract key name (everything before the first =)
  key="${line%%=*}"
  key="${key#"${key%%[![:space:]]*}"}"  # trim leading whitespace
  key="${key%"${key##*[![:space:]]}"}"  # trim trailing whitespace
  KEYS+=("$key")
done < .env.example

if [ ${#KEYS[@]} -eq 0 ]; then
  echo "❌ No keys found in .env.example."
  exit 1
fi

echo "Found ${#KEYS[@]} key(s) in .env.example:"
for k in "${KEYS[@]}"; do
  echo "  - $k"
done
echo ""

# Collect values
rm -f .env
for k in "${KEYS[@]}"; do
  read -rp "  Enter value for $k (leave blank to skip): " val
  if [ -n "$val" ]; then
    echo "${k}=${val}" >> .env
  fi
done

if [ ! -f .env ]; then
  echo "⚠️  No values entered. Nothing to encrypt."
  exit 0
fi

# The keyring must be UNLOCKED before `secret-tool store`, not during it. A
# locked collection makes the store issue a D-Bus unlock Prompt, and with no
# graphical prompter attached to this terminal that prompt is never answered --
# the call blocks forever with nothing on screen. Unlock first, with stderr
# visible, so a failure is a message rather than a hang.
echo "🔓 Unlocking the OS keyring (you may be prompted)..."
if ! secret-tool search --unlock service pricing-perspective >/dev/null 2>&1; then
  : # a miss is expected on first run; the --unlock side effect is the point
fi

echo ""
echo "🔒 Encrypting .env (dotenvx >=1 encrypts values in place)..."
dotenvx encrypt

# dotenvx 2.x writes the private key here and encrypts .env in place; it does
# NOT produce a .env.vault. This file is the ONLY copy of the key until the
# keyring store below succeeds -- losing it makes the encrypted .env
# permanently unreadable, so nothing may delete it before then.
if [ ! -f .env.keys ]; then
  echo "❌ .env.keys was not generated. Something went wrong."
  exit 1
fi

PRIVATE_KEY=$(grep 'DOTENV_PRIVATE_KEY' .env.keys | head -1 | cut -d= -f2-)

echo "🔑 Storing decryption key in OS keyring..."
# stderr is deliberately NOT discarded: swallowing it is what made an unlock
# failure look like a silent hang.
if ! printf '%s' "$PRIVATE_KEY" | secret-tool store \
  --label="pricing-perspective DOTENV_PRIVATE_KEY" \
  service pricing-perspective \
  key DOTENV_PRIVATE_KEY; then
  echo ""
  echo "❌ Could not store the key in the keyring."
  echo "   .env.keys has been LEFT IN PLACE deliberately — it is the only copy"
  echo "   of the key that decrypts .env. Back it up before retrying."
  exit 1
fi

# Verify the round trip before destroying the on-disk copy. Only a readback
# proves the key is recoverable; a zero exit from `store` does not.
if [ "$(secret-tool lookup service pricing-perspective key DOTENV_PRIVATE_KEY)" != "$PRIVATE_KEY" ]; then
  echo ""
  echo "❌ Keyring readback did not match. .env.keys LEFT IN PLACE."
  exit 1
fi

# Safe from here: the key is in the keyring and verified readable.
KEY_SECURED=1
echo "🧹 Cleaning up the on-disk key..."
rm -f .env.keys

echo ""
echo "✅ Done! Your secrets are encrypted in place in .env."
echo "   The decryption key is stored in your OS keyring."
echo ""
echo "   Next steps:"
echo "     direnv allow          # activate the shell"
echo "   NOTE: .env is gitignored here (.gitignore:102). The encrypted file is"
echo "   safe to commit, but un-ignoring it is a policy call, not this script's."
echo ""
