FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app

# системные зависимости
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# python-зависимости
RUN pip install --no-cache-dir \
    matplotlib \
    seaborn \
    scikit-learn

# копируем код
COPY train.py /app/train.py

CMD ["python", "train.py"]
