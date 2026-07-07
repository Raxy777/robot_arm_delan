"""Physical parameters of the 2-link planar arm.

These are the single source of truth. The MuJoCo XML (model/arm2.xml) is
built to match these exactly, so the analytic dynamics in dynamics.py and the
simulator agree. If you change anything here, regenerate/adjust the XML too
(or just run verify_dynamics.py to confirm they still match).

Convention
----------
- Arm lives in the x-y plane. x points right, y points up.
- Gravity acts along -y (magnitude G).
- Both joints are revolute (hinge) about the +z axis (out of the page).
- q1 is measured from the +x axis to link 1; q2 is the relative angle of
  link 2 with respect to link 1 (standard elbow convention).
- Each link is a uniform thin rod: center of mass at its geometric center,
  inertia about the COM (z-axis) = (1/12) m L^2.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArmParams:
    # link masses (kg)
    m1: float = 1.0
    m2: float = 1.0
    # link lengths (m)
    l1: float = 1.0
    l2: float = 1.0
    # distance from joint to link center of mass (m)
    lc1: float = 0.5
    lc2: float = 0.5
    # link inertia about its own COM, z-axis (kg m^2); thin rod = m L^2 / 12
    I1: float = 1.0 * 1.0 ** 2 / 12.0
    I2: float = 1.0 * 1.0 ** 2 / 12.0
    # gravity (m/s^2)
    g: float = 9.81


PARAMS = ArmParams()
