# Deployment guide

The repository now includes `render.yaml` and `Dockerfile` for deployment.

## Fastest full deployment: Render Blueprint

1. Create or sign in to a Render account: https://render.com
2. Select **New + → Blueprint**.
3. Connect GitHub and select `prasoonsingh225-beep/skyguard-aws`.
4. Render detects `render.yaml` and creates:
   - `aero-guard-api`: FastAPI inference service
   - `aero-guard-dashboard`: Streamlit dashboard
5. Click **Apply** and wait for both services to become live.
6. Open the dashboard service URL. The API Swagger page is available at `/docs`.

The first build trains a small demonstration model. For production, replace the synthetic training data with validated historical AWS data and use a persistent model artifact.

## Docker deployment

```bash
docker build -t aero-guard .
docker run --rm -p 8000:8000 aero-guard
```

Open `http://localhost:8000/docs`.

## Important limitation

A public cloud URL cannot be created from this repository alone. Render requires the repository owner to authorize the deployment and accept the provider's account terms. After the Blueprint is deployed, Render supplies the final public URLs.
