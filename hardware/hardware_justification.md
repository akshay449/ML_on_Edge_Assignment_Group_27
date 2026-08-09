# Hardware Justification

## Overview

LogiEdge must run on a truck-mounted edge node performing continuous sensor
monitoring, local ML inference, MQTT messaging, and Docker/Ansible-managed OTA
updates — powered from the truck's 12 V auxiliary supply via a DC-DC converter within
a 10 W AI power budget.

---

# 1. Candidate Hardware (per project brief, Task B1)

| Option | Hardware | India Price | TDP |
|---|---|---|---|
| 1 | Raspberry Pi 5 (8 GB) + AI HAT+ (13 TOPS Hailo-8L) | ~₹15,000/truck | 7.5 W |
| 2 | Jetson Orin Nano Super Dev Kit (67 TOPS) | ~₹45,000/truck | 15 W (moderate load) |
| 3 | STM32H7-based custom MCU + sensor ICs | ~₹3,500/truck | 0.4 W |

Fleet cost at scale (85-truck pilot → 265-truck full fleet):

| Option | Pilot (85 trucks) | Full fleet (265 trucks) |
|---|---|---|
| Pi 5 + AI HAT+ | ₹12,75,000 | ₹39,75,000 |
| Jetson Orin Nano | ₹38,25,000 | ₹1,19,25,000 |
| STM32H7 MCU | ₹2,97,500 | ₹9,27,500 |

---

# 2. Constraint Triangle Analysis

The relevant vertices are **Cost**, **Power**, and **Compute/Latency capability**,
with the project's own mandatory deliverables (Docker containerisation, local
Mosquitto broker, Ansible-managed deployment) acting as a fourth, non-negotiable
**software-platform** constraint.

- **Compute/Latency:** the trained model (M1: 32→16→3 MLP) is tiny — well under 2 ms
  per inference on a plain CPU. None of the three options are latency-constrained
  against the 90-second SLA; even the STM32H7 clears it trivially. AI-accelerator TOPS
  (13 or 67) are irrelevant here — this workload never approaches needing hardware
  acceleration.
- **Power:** all three options fit inside the 10 W budget (0.4 W, 7.5 W, 15 W-peak
  respectively), though the Jetson's 15 W *moderate-load* figure leaves the least
  headroom against the ceiling once sensors, radios, and background OS load are
  added.
- **Cost at fleet scale:** this is the dominant vertex. Jetson Orin Nano costs 3× the
  Pi 5 option and 13× the STM32 option with zero latency or accuracy benefit for a
  model this small — it is dominated on every axis and is rejected outright.
- **Software platform (the deciding constraint):** Task D2/E2 mandate a Docker
  container running the inference service and an Ansible playbook managing
  deployment over SSH. Both require a full Linux userspace, a filesystem, and a
  Python runtime with pip-installable packages. The STM32H7 is a bare
  microcontroller — it cannot run Docker, Ansible, or a local Mosquitto broker
  without a companion Linux gateway, which erases its cost/power advantage the
  moment that gateway is added to the bill of materials.

**Verdict:** Raspberry Pi 5 (the AI HAT+ accelerator is not required by this
workload but is a low incremental cost and headroom for future model growth) is the
selected platform. It clears the 90 s latency SLA by more than an order of magnitude,
sits comfortably inside the 10 W budget, satisfies every mandated software
deliverable natively, and remains affordable at fleet scale (₹39.75 lakh for all 265
trucks — against a single documented ₹28 lakh spoilage incident, the ROI case is
immediate). The Jetson Orin Nano is rejected for cost and power with no offsetting
benefit; the STM32H7 is rejected because it cannot host the required software stack
without an added gateway device.

---

# 3. Arithmetic Intensity and Roofline Analysis (Task B2)

Given: model performs **45 MFLOPs/inference**, accesses **18 MB** of data
(weights + activations) per inference. Raspberry Pi 5 CPU: **16 GFLOP/s** (NEON SIMD),
**12 GB/s** LPDDR4X bandwidth.

**Arithmetic Intensity (AI):**
AI = FLOPs / Bytes = 45 × 10⁶ / 18 × 10⁶ = **2.5 FLOPs/byte**

**Ridge point:**
Ridge = Peak Compute / Peak Bandwidth = 16 GFLOP/s / 12 GB/s = **1.33 FLOPs/byte**

**Classification:** AI (2.5) > Ridge (1.33) → the model sits to the right of the ridge
point on the roofline: **compute-bound**, not memory-bandwidth bound.

**Implication:** because the model is compute-bound, optimisations that reduce memory
traffic (better caching, weight reuse) will not move the needle much. The effective
levers are the ones that reduce FLOPs or increase effective throughput per FLOP:
structured pruning (fewer multiply-accumulates per inference — this is exactly what
M3 does), and INT8 quantisation, which on NEON SIMD raises effective GFLOP/s
throughput for the same silicon compared to FP32. This is consistent with the
project's M2/M3 optimisation path already being the right lever, rather than
prioritising data-layout/caching changes.

---

# 4. Sensors

| Sensor | Purpose | Sampling Rate |
|---|---|---|
| Temperature | Refrigerated storage temperature | 1 Hz |
| Vibration | Refrigeration/compressor mechanical anomaly | 0.5 Hz |
| Door | Open/close event detection | Discrete (event-driven) |

---

# 5. Communication

MQTT (Mosquitto broker, local to the truck) connects the simulator, consumer, and
inference stages. Raw telemetry never leaves the truck; only inference results and
alerts cross the cellular uplink (see `constraint_analysis.md` §3 for the bandwidth
math this enables).

---

# 6. Resource Budget

- **Memory:** inference container target < 64 MB RAM; model binary target < 10 KB
  (INT8, post-pruning) so it would also fit microcontroller flash if the workload
  were ever ported down.
- **CPU:** moderate utilisation during training (off-truck, one-time); low
  utilisation during on-truck inference.
- **Power:** continuous operation within the 10 W AI budget, 7.5 W TDP for the
  selected board.

---

# Conclusion

The Raspberry Pi 5 + AI HAT+ satisfies the Constraint Triangle for the FreightBridge
pilot: it clears the latency SLA by a wide margin, sits within the power budget, is
affordable at 265-truck fleet scale, and is the only one of the three candidates that
natively supports the project's mandatory Docker/MQTT/Ansible software stack without
an additional gateway device.
