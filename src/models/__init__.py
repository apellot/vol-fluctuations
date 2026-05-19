"""Set-based architectures for centrality from per-track features.

All four architectures share the same I/O contract — defined in src/models/heads.py:
  input :  cont (B, L, N_CONT)   pT, η_lab, φ, charge
           mask (B, L)           True where a real particle sits
           event_feats (B, F_e)  √sNN, mult_lab, mean_pT_lab, total_pT_lab
  output:  dict with mu, nu, alpha, beta  (NIG b head)  and  logits  (centrality classifier)

Differences are purely in how per-particle info is aggregated into an event embedding:
  mlp_pool        — mean-pool then dense
  deepsets        — learned φ, masked pool, learned ρ
  set_transformer — SAB self-attention, PMA pooling
  efn             — Komiske-Metodiev-Thaler Energy Flow Network (pT-weighted latent)
"""
