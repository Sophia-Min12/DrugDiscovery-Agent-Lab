"""DrugDiscovery-Agent-Lab: a drug-discovery agent that checks its own science.

chem / drugs   deterministic cheminformatics, every method-dependent value named
claims         claims, evidence, the five verdicts
extract        free text -> claims (rule-based, auditable)
verifier       claims -> verdicts, by recomputation with a tool
omics          RNA-seq differential expression on planted truth
gnn            solubility GNN with conformal intervals and an applicability domain
tools          the registry shared by the agent and the MCP server
agent          the LangGraph plan -> act -> draft -> verify -> revise -> report loop
mcp_server     the tools and the verifier over MCP
sandbox        Docker sandbox for model-written code
"""
