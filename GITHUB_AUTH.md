# GitHub Authentication Guide

## Issue: Permission Denied (403 Error)

You need to authenticate with GitHub. Here are your options:

## Option 1: Use Personal Access Token (Recommended)

### Step 1: Create a Personal Access Token

1. Go to GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "Leukemia Project")
4. Select scopes: **repo** (full control of private repositories)
5. Click "Generate token"
6. **Copy the token immediately** (you won't see it again!)

### Step 2: Use Token for Push

When prompted for password, use the token instead:

```bash
git push -u origin main
# Username: AlphaDoc1
# Password: <paste your token here>
```

## Option 2: Update Remote URL with Token

```bash
git remote set-url origin https://AlphaDoc1:<YOUR_TOKEN>@github.com/AlphaDoc1/canser_detection-.git
git push -u origin main
```

## Option 3: Use GitHub CLI

```bash
# Install GitHub CLI if not installed
# Then authenticate:
gh auth login

# Then push:
git push -u origin main
```

## Option 4: Use SSH (Alternative)

### Step 1: Generate SSH Key
```bash
ssh-keygen -t ed25519 -C "hemanthsavanth5555@gmail.com"
```

### Step 2: Add SSH Key to GitHub
1. Copy your public key: `cat ~/.ssh/id_ed25519.pub`
2. Go to GitHub → Settings → SSH and GPG keys → New SSH key
3. Paste the key and save

### Step 3: Change Remote URL
```bash
git remote set-url origin git@github.com:AlphaDoc1/canser_detection-.git
git push -u origin main
```

## Quick Fix (Easiest)

If you have GitHub Desktop installed, you can:
1. Open GitHub Desktop
2. Add the repository
3. Push from there

Or use the Personal Access Token method (Option 1) - it's the quickest!


