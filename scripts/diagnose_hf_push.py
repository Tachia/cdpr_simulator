"""Definitive diagnostic for the 'pre-receive hook declined' HF push error.

Calls Hugging Face's REST API directly with a token you provide, so we
can compare the token's REAL account against the Space's REAL owner.
Uses only stdlib urllib --- no pip install required.

Usage::

    # Token in environment (preferred --- never echoed):
    $env:HF_TOKEN = "hf_..."
    python scripts\diagnose_hf_push.py

    # Or pass it explicitly (not recommended; ends up in shell history):
    python scripts\diagnose_hf_push.py --token hf_...

The output answers four questions:

    1. Is the token valid?
    2. Which HF account does it belong to?
    3. Does spaces/JoeTach/cdpr-simulator actually exist?
    4. Does the token's account have write access to it?

If any of the four is wrong, the script prints the exact next step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


HF_API = "https://huggingface.co/api"
SPACE_OWNER = "JoeTach"
SPACE_NAME = "cdpr-simulator"


def _get_json(url: str, *, token: str | None = None,
              timeout: float = 15.0) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def whoami(token: str) -> dict[str, Any]:
    """Resolve the token to its actual HF account.

    Returns the parsed /api/whoami-v2 response on success. Raises with
    the HTTP status code on failure (commonly 401 for invalid tokens).
    """
    return _get_json(f"{HF_API}/whoami-v2", token=token)


def space_info(owner: str, name: str) -> dict[str, Any]:
    """Fetch public Space metadata. Returns {} on 404.

    Note: the Spaces metadata endpoint returns 404 for both
    nonexistent and private Spaces, so a 404 isn't conclusive proof
    the Space doesn't exist --- but combined with the whoami result
    it's usually enough to diagnose."""
    try:
        return _get_json(f"{HF_API}/spaces/{owner}/{name}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return {}
        raise


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diagnose HF push authorisation.")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN", ""),
                   help="HF write-scope token. Reads $HF_TOKEN by default.")
    p.add_argument("--owner", default=SPACE_OWNER)
    p.add_argument("--space", default=SPACE_NAME)
    args = p.parse_args(argv)

    token = args.token.strip()
    if not token:
        print("ERROR: no token provided.", file=sys.stderr)
        print("Run one of:")
        print("  $env:HF_TOKEN = 'hf_...'; python scripts\\diagnose_hf_push.py")
        print("  python scripts\\diagnose_hf_push.py --token hf_...")
        return 2

    print("=" * 60)
    print("HF push diagnostic")
    print("=" * 60)
    print(f"Token       : hf_...{token[-6:]}  (last 6 chars only)")
    print(f"Target      : {args.owner}/{args.space}")
    print()

    # Step 1: validate the token by asking who it belongs to.
    print("[1/3] Calling /api/whoami-v2 with your token ...")
    try:
        me = whoami(token)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print()
            print("RESULT: Token is INVALID or revoked (HTTP 401).")
            print()
            print("Next step:")
            print("  1. Open https://huggingface.co/settings/tokens")
            print("  2. Generate a NEW token with Type = Write")
            print("  3. Re-run this script with the new token")
            return 3
        print(f"  ERROR: HTTP {exc.code} from HF API: {exc.reason}")
        return 4
    except Exception as exc:
        print(f"  ERROR: could not reach HF API: {exc}")
        return 4

    token_user = me.get("name") or ""
    token_type = me.get("type") or "?"
    token_orgs = [o.get("name") for o in (me.get("orgs") or [])
                  if isinstance(o, dict)]
    print(f"  -> token belongs to user : {token_user!r}")
    print(f"  -> token type            : {token_type!r}")
    print(f"  -> orgs the user is in   : {token_orgs}")
    print()

    # Step 2: check the Space exists and who owns it.
    print(f"[2/3] Looking up Space {args.owner}/{args.space} ...")
    info = space_info(args.owner, args.space)
    if not info:
        print("  -> Space is NOT publicly visible (404 from /api/spaces/...).")
        print("     This could mean:")
        print("       (a) the Space genuinely doesn't exist yet,")
        print("       (b) it exists but is private, or")
        print("       (c) the owner/name in the URL is misspelled.")
        space_author = ""
    else:
        space_author = info.get("author") or (info.get("id") or "/").split("/")[0]
        is_private = info.get("private", False)
        sdk = info.get("sdk", "?")
        print(f"  -> Space exists         : YES")
        print(f"  -> Space author / owner : {space_author!r}")
        print(f"  -> private              : {is_private}")
        print(f"  -> SDK                  : {sdk}")
    print()

    # Step 3: compare and verdict.
    print("[3/3] Verdict")
    print("-" * 60)
    if token_user == args.owner:
        if info:
            print(f"GOOD: token user '{token_user}' matches Space owner.")
            print()
            print("The push SHOULD work. If it doesn't, the most likely cause is")
            print("that the token does not have Write scope (it might be Read or")
            print("a fine-grained token without the right Space granted). Open:")
            print("  https://huggingface.co/settings/tokens")
            print("and confirm the token's row says 'Type: Write'. If not, create")
            print("a fresh Write-scope token and retry.")
            return 0
        else:
            print(f"PROBABLE FIX: the Space '{args.owner}/{args.space}' does")
            print("not appear in the public listing, but your token does belong")
            print("to that username. Either:")
            print(f"  (a) create the Space at https://huggingface.co/new-space")
            print(f"      with name '{args.space}' under owner '{args.owner}'")
            print(f"      using SDK = Gradio, hardware = CPU basic, then retry")
            print(f"      the push; or")
            print(f"  (b) check the spelling --- maybe it's actually a different")
            print(f"      Space name.")
            return 5

    # token_user != args.owner --- the actual mismatch case.
    print(f"ROOT CAUSE FOUND.")
    print()
    print(f"  The token you used belongs to HF account : {token_user!r}")
    print(f"  But you're trying to push to             : {args.owner}/{args.space}")
    print()
    print(f"  HF rejects the push because '{token_user}' has no write")
    print(f"  permission on Spaces owned by '{args.owner}'.")
    print()
    print("Three ways to fix this --- pick ONE:")
    print()
    if args.owner in token_orgs:
        print(f"  A. (best if '{args.owner}' is an organisation you belong to)")
        print(f"     Your account '{token_user}' is a member of org")
        print(f"     '{args.owner}'. Check on HF:")
        print(f"     https://huggingface.co/organizations/{args.owner}/members")
        print(f"     If your role is 'read', ask the org admin to promote you")
        print(f"     to 'write' or 'admin', then retry the push.")
        print()
    print(f"  B. Log into HF as '{args.owner}' and generate a token there.")
    print(f"     Open https://huggingface.co/login, sign in as '{args.owner}',")
    print(f"     then go to https://huggingface.co/settings/tokens to make a")
    print(f"     new Write-scope token. Retry the push with THAT token.")
    print()
    print(f"  C. Push to a Space '{token_user}' actually owns instead.")
    print(f"     Create a new Space at https://huggingface.co/new-space")
    print(f"     with owner '{token_user}' and name 'cdpr-simulator'")
    print(f"     (SDK = Gradio, hardware = CPU basic), then retarget the")
    print(f"     remote and push:")
    print(f"         git remote set-url space \\\\")
    print(f"             https://huggingface.co/spaces/{token_user}/cdpr-simulator")
    print(f"         git push --force \\\\")
    print(f"             https://{token_user}:<TOKEN>@huggingface.co/spaces/{token_user}/cdpr-simulator \\\\")
    print(f"             main:main")
    print()
    return 6


if __name__ == "__main__":
    sys.exit(main())
