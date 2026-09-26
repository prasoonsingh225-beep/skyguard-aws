FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN python data_generator.py --out data.csv --n 8000 --spikes 8 --stuck 3 --drift 2 --comm 2 \
    && python train.py --data data.csv --model model.h5 --window 60 --epochs 2 --batch 128

EXPOSE 8000
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
