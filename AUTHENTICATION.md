# HuggingFace Authentication for Finnish TTS Training

This document explains how to securely authenticate with HuggingFace to download the Finnish TTS model.

## Why Authentication?

The base model (`fishaudio/openaudio-s1-mini`) is a **gated model** on HuggingFace, meaning:
- You need to accept the model's license terms first
- Authentication proves you've accepted the terms
- Without authentication, training will fail when trying to download the model

## Security First 🔒

We support **three secure methods** (in order of security):

### Method 1: Interactive Login (MOST SECURE) ⭐

This method stores your token securely in `~/.cache/huggingface/token` and never exposes it in process lists.

```bash
huggingface-cli login
```

You'll be prompted interactively:
```
    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|
    _|    _|  _|    _|  _|        _|          _|    _|  _|  _|
    _|    _|  _|    _|  _|        _|          _|    _|    _|_|
    _|      _|_|_|    _|_|_|    _|_|_|    _|_|_|  _|      _|

To authenticate, you need to provide your token. Tokens are available via https://huggingface.co/settings/tokens

Token:
```

Paste your token and press Enter. Done! No token appears in:
- Environment variables
- Shell history
- Process lists (`ps aux`)
- `.env` files
- Git history

**This is the recommended method for production Launchables.**

---

### Method 2: Environment Variable (CI/CD Friendly)

Use this method for automated deployments or CI/CD pipelines.

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxx"
bash setup.sh
```

**Security notes:**
- Token is passed securely via stdin to Python, not as a command-line argument
- It's still visible in shell history and environment (`echo $HF_TOKEN` will show it)
- Best for automated systems with secure secret management
- **Never paste your real token in code or commit it**

---

### Method 3: .env File (Local Development)

For local development only.

1. **Copy the template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` and add your token:**
   ```bash
   # .env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxx
   ```

3. **Run setup:**
   ```bash
   bash setup.sh
   ```

**Security notes:**
- `.env` is gitignored (never committed to repo)
- Only use for local development
- Token is sourced into the current shell session
- Visible in your local filesystem

---

## Getting Your Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Set permissions to **"Read"** (model download only)
4. Give it a name like "Finnish TTS Training"
5. Copy the token (starts with `hf_`)

## Accepting the Model License

After getting a token, you must **accept the model license:**

1. Visit: https://huggingface.co/fishaudio/openaudio-s1-mini
2. Click **"Access repository"** or **"Accept"**
3. Follow the prompts to accept the license terms
4. Once approved, the model will be available for download

---

## Troubleshooting

### "Access denied" or "Model not found"

```
gated_repo_error: You don't have access to this model
```

**Solution:**
1. Check you accepted the license: https://huggingface.co/fishaudio/openaudio-s1-mini
2. Wait 5-10 minutes for approval (sometimes instant)
3. Try authenticating again:
   ```bash
   huggingface-cli logout
   huggingface-cli login
   ```

### "Invalid token"

```
HTTPError: 401 Client Error: Unauthorized
```

**Solution:**
1. Generate a new token: https://huggingface.co/settings/tokens
2. Make sure it has **"Read"** permission
3. Try `huggingface-cli login` again

### Token in shell history

If you used `export HF_TOKEN=xxx` and it's in your history:

```bash
# Bash: Edit ~/.bash_history
# Zsh: Edit ~/.zsh_history

# Then regenerate a new token to be safe:
# https://huggingface.co/settings/tokens (Delete the old one)
```

---

## Production Deployment (Brev Launchables)

For community sharing via Brev Launchables:

1. **Users provide their own token** - never hard-code in the repo
2. **Recommended flow:**
   - User runs: `huggingface-cli login` on their Brev instance
   - User runs: `bash setup.sh`
   - setup.sh detects existing authentication automatically

3. **For automated deployments:**
   - Pass `HF_TOKEN` as an environment variable at deployment time
   - The script handles secure authentication

---

## FAQ

**Q: Can I share my token?**  
A: **No, never!** Your token gives access to your HuggingFace account. Anyone with your token can:
- Download your private models
- Modify your account settings
- Upload files to your repos

**Q: What if my token is leaked?**  
A: Immediately delete it on https://huggingface.co/settings/tokens and create a new one.

**Q: Does the script store my token?**  
A: No. The script:
1. Uses Method 1 (interactive login) → Token stored in `~/.cache/huggingface/token` (by HF's official tool)
2. Uses Method 2 (env var) → Token passed to Python via stdin (never stored)
3. Uses Method 3 (.env) → Only used in current session

**Q: Do I need to login every time?**  
A: No. After the first login with Method 1, you're authenticated for future sessions.

**Q: Can I use my HuggingFace API key?**  
A: Yes, but it's not recommended. Use a **token** instead (different thing). API keys are for API endpoints, tokens are for model/dataset downloads.

---

## Security Best Practices

✅ **DO:**
- Use `huggingface-cli login` for interactive sessions
- Create a token with minimal permissions (just "Read")
- Accept the model license before deployment
- Delete tokens after deployment
- Use environment variables for CI/CD with proper secret management

❌ **DON'T:**
- Commit `.env` files to Git
- Hard-code tokens in scripts or notebooks
- Share your token with anyone
- Use tokens with "Write" or "Admin" permissions
- Check tokens into version control

---

Need help? Check the [README](README.md) or file an issue on GitHub!
