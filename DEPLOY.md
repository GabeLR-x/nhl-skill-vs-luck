# Deploying the dashboard to Streamlit Community Cloud

The app needs nothing beyond what's in git: it builds its tables from the three
tracked CSVs in `data/` on startup (cached), so `nhl.db` is not required.

## One-time setup

1. **Push `main` to GitHub** (from VS Code as usual).
2. Go to <https://share.streamlit.io> and **sign in with GitHub**. Approve the
   permissions it asks for. To deploy from a private repo, it needs access to
   private repositories too.
3. Click **Create app** → **Deploy a public app from GitHub** (or "Yup, I have an app").
4. Fill in:
   | Field | Value |
   |---|---|
   | Repository | `GabeLR-x/nhl-skill-vs-luck` |
   | Branch | `main` |
   | Main file path | `dashboard/app.py` |
   | App URL | pick a subdomain, e.g. `nhl-skill-vs-luck` |
5. Open **Advanced settings** and set **Python version: 3.12**. No secrets are needed.
6. Click **Deploy**. The first build takes a few minutes (it installs the pinned
   `requirements.txt`); later restarts are faster.

## Make it viewable by recruiters

An app deployed from a **private** repo is private: only people you invite can
open it. For a link you can put on a resume, either:

- make the GitHub repo public (Settings → General → Danger Zone → Change visibility), or
- keep the repo private and, in the app's **Settings → Sharing**, make the app public.

Then open the app in a private browser window to confirm it loads without a login.

## After deploying

- Paste the app URL into the "Live demo" line at the top of `README.md`.
- Every push to `main` redeploys automatically.
- If something breaks: open the app → **Manage app** (bottom right) → read the
  logs, or **Reboot app**. Free apps go to sleep after a few days without
  visitors; the first visitor clicks "wake up" and waits about 30 seconds.

## Run it locally exactly like the cloud does

```bash
git clone https://github.com/GabeLR-x/nhl-skill-vs-luck.git
cd nhl-skill-vs-luck
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Run it from the repo root so Streamlit picks up the dark theme in `.streamlit/config.toml`.
