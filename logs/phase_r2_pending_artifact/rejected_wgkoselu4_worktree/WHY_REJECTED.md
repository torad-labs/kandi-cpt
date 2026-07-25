# Rejected in-tree r3 implementation (run wgkoselu4, written 05:45, violating the no-repo-writes law)

Preserved for the record before reverting the working tree to HEAD. Rejected ON MEASUREMENT by the
round-2 implementer (run w4oe56arp), not on preference:

- Its own docstring claims entity additions become "+0.0000". Measured: shadow_obj -0.0114,
  shadow_subj -0.0217 (subject_in_unit still reads the live entity table — one of two entity
  levers closed, both advertised closed).
- Genuine-gain sensitivity cut 3.4x (+0.0038 under its 267-denominator vs +0.0128 under 78).
