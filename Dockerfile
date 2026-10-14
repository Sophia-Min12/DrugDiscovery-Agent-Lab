# The agent's tools and verifier as a container - by default the MCP server on stdio:
#
#   docker build -t drugagent .
#   docker run -i --rm drugagent                                    # MCP over stdio
#   docker run --rm drugagent python -m drugagent verify "Aspirin has a TPSA of 83.6."
#   docker run --rm -e OLLAMA_HOST=http://host.docker.internal:11434 drugagent \
#       python -m drugagent ask "Summarise the drug-likeness of imatinib."
FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir numpy scipy pandas matplotlib rdkit langgraph mcp pytest

# Run from source, so data/ and models/ sit where the package expects them.
COPY drugagent ./drugagent
COPY data/approved_drugs.csv ./data/approved_drugs.csv
COPY models ./models
COPY tests ./tests
COPY pyproject.toml ./

ENV PYTHONPATH=/app MPLCONFIGDIR=/tmp PYTHONUNBUFFERED=1
USER 65534:65534
CMD ["python", "-m", "drugagent.mcp_server"]
