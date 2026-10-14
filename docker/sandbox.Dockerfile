# The image generated analysis code runs in: the scientific stack and nothing else.
# Every run adds --network none, a read-only root, no capabilities and user nobody.
FROM python:3.12-slim

RUN pip install --no-cache-dir numpy scipy pandas matplotlib rdkit

WORKDIR /work
USER 65534:65534
CMD ["python", "main.py"]
