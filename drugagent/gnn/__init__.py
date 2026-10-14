"""A message-passing GNN for aqueous solubility, in plain PyTorch.

No graph library: message passing is `index_add_` over an edge list, which keeps
every step of the model visible. The point of this subpackage is less the model than
what an agent may say about its output - a prediction is a claim with an interval
and an applicability domain, and the verifier treats it as one.
"""
