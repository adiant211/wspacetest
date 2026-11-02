Saved all weights. The network implements handcrafted liberty propagation, guarded captures, atari escapes, self-atari avoidance, eye preservation, and connection/shape preferences.

Across both tests of the final weights, it won:
- **79.7%** against strong (153/192)
- **95.8%** against weak (46/48)
- **100%** against random (16/16)

Its tactical discipline and connected shapes should generalize, though hidden-opponent performance remains uncertain. No training, search, or sandbox forward evaluations were used.