# Deploying the explorer (free)

## Streamlit Community Cloud (recommended)

1. Repo is on GitHub (done).
2. Go to https://share.streamlit.io, sign in with GitHub, click **New app**.
3. Repository: `ashishlandiwal/ecommerce-customer-segmentation`, branch `main`,
   main file path: `app/streamlit_app.py`.
4. **Deploy.** On first load the app generates the segments and serves at
   `https://<your-app>.streamlit.app`.
5. Paste that URL into the repo's **About → Website** and onto your resume/LinkedIn.

## Local

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
