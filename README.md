## Overview

This repository provides the source code accompanying the paper
**“Information-Conformant Modeling and Control for Linear and Nonlinear Systems.”**

The paper proposes an information-conformant, model-based framework for set-based
system identification of linear and nonlinear systems with unknown dynamics and
uncertain disturbances. The approach constructs bounded families of admissible
models that are consistent with both measured data (posterior information) and
available side information (prior knowledge).

The identification problem is formulated as a convex optimization program that
simultaneously learns matrix zonotope representations of the uncertain system
dynamics and zonotopic disturbance bounds, while discarding models that are not
conformant with the available information. The resulting information-conformant
models (ICMs) are compared against data-conformant models that ignore side
information, demonstrating improved feasibility, modeling accuracy, and reduced
conservativeness.

For linear systems, the learned ICMs are used to synthesize controllers that
robustly enforce the $\lambda$-contractiveness of a polyhedral set, guaranteeing
invariance and stability under model uncertainty and bounded disturbances.
For structured affine nonlinear systems, a semidefinite-program-based controller
design is employed to robustly stabilize the closed-loop system.
