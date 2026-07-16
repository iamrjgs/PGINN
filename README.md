Physics and Geometry Informed Neural Network (PGINN) VMAT Radiotherapy Dose Engine

Uses deep learning to perform fast dose calculations of patient VMAT plans.

Implements GPU-accelerated ray-tracing capabilities to generate input "hand-calc" doses for control points,
based on patient treatment planning CT, RTPLAN, RTSTRUCTS, RTDOSE, and beam PDD and OAR data.

Uses physical density and ray-traced doses as inputs for deep learning models to predict individual control point dose
distributions, then combines doses from all control points to generate full-arc and full-plan dose distributions.

Deep learning models were trained on Monte-Carlo generated dose distributions.