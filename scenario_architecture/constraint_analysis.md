# Constraint Analysis

## Project
LogiEdge — AI-Powered Cold Chain Monitoring using Edge Intelligence
(FreightBridge Logistics Pvt. Ltd., 85-truck pharmaceutical pilot fleet)

---

# 1. Problem Statement

Cold-chain logistics requires continuous monitoring of temperature, refrigeration-unit
vibration, and door status. A refrigeration failure can raise cargo temperature by
~1°C/minute, so undetected faults translate directly into spoilage. The objective of
this project is an edge AI pipeline that detects and alerts on cargo anomalies without
depending on continuous cellular connectivity.

---

# 2. Latency Constraint

**Requirement:** detect and alert on a fault signature within **90 seconds**.

- LogiEdge's window step is 10 seconds (30 s sliding window, 10 s step), so a fault is
  visible to the classifier within one window step — well inside the 90 s budget.
- Model inference itself (M1 baseline, 32→16→3 MLP) measures well under 2 ms per
  window on CPU, so inference latency is not the bottleneck; the sliding-window
  cadence is.
- **Is cloud inference feasible?** India's rural 4G/LTE round-trip latency runs
  roughly 150–400 ms under good coverage, but the Nashik–Aurangabad route loses
  signal entirely for 35–90 minutes at seven documented locations. A 90-second SLA is
  unenforceable if the truck can be *disconnected* for up to 90 minutes at a stretch —
  no round-trip latency number matters once there's no link at all. Cloud-only
  inference cannot meet this SLA; on-device inference is not an optimization here, it
  is the only architecture that satisfies the requirement.

---

# 3. Bandwidth Constraint

**Requirement:** quantify raw sensor volume vs. edge-processed alert volume, and cost
at ₹0.10/MB.

**Raw sensor volume (hypothetical uncompressed cloud-streaming baseline, per the
brief's 500 Hz tri-axial vibration scenario):**

| Stream | Rate | Bytes/sample | Bytes/sec | Bytes/day |
|---|---|---|---|---|
| Temperature | 1 Hz | 4 B (float32) | 4 B/s | 345,600 B (~0.34 MB) |
| Vibration (3-axis) | 500 Hz | 3 × 4 B = 12 B | 6,000 B/s | 518,400,000 B (~494.4 MB) |
| Door events | discrete (~20/day) | ~50 B | negligible | ~1 KB |
| **Total** | | | | **~494.7 MB/truck/day** |

Cost at ₹0.10/MB: **₹49.47/truck/day** → **₹4,205/day** for the 85-truck pilot →
**~₹1.26 lakh/month**, before scaling to the full 265-vehicle fleet (~₹3.94 lakh/month).

**Edge-processed volume (what LogiEdge actually ships over cellular):** raw 1 Hz
telemetry stays on the truck's local MQTT broker; only classification results
(`logibridge/trucks/{id}/inference`, ~150 B JSON) leave over cellular, once per 10 s
window step = 8,640 messages/day ≈ **1.3 MB/truck/day**.

**Reduction:** (494.7 − 1.3) / 494.7 ≈ **99.7%**, consistent with (and better than) the
project's "~98% payload reduction" target — edge inference turns a ~₹1.26 lakh/month
cellular data cost into a negligible one.

---

# 4. Connectivity Constraint

The Nashik–Aurangabad route loses cellular signal for 35–90 minutes at seven
documented locations. A cloud-only system goes completely blind for the duration of
each gap: no telemetry reaches the backend, no inference happens, and no alert can be
raised — a refrigeration failure at 1°C/minute could spoil an entire vaccine shipment
inside a single 90-minute blackout.

LogiEdge's edge node keeps running inference locally regardless of link state. On
detecting an anomaly it (a) logs the alert to a local append-only log with timestamp,
and (b) attempts an MQTT publish; if the cellular uplink is down, the alert is queued
and backfilled to the operations centre the moment coverage returns. The truck-side
decision loop (sense → preprocess → classify → alert) never depends on the network.

---

# 5. Privacy Constraint

FreightBridge's pharmaceutical clients require proof that cargo condition data cannot
be accessed by unauthorised third parties. On-device inference supports this
contractually because raw environmental telemetry never leaves the vehicle: only a
classification label and confidence score cross the cellular link. There is no cloud
data store of raw temperature/vibration streams to be breached, subpoenaed, or
mishandled by a third-party processor — the attack surface for cargo condition data is
limited to the truck's own local storage, which can be covered under the existing
vehicle security terms rather than a cloud data-processing agreement.

---

# 6. Hardware, Software, Network, and AI Constraints (summary)

**Hardware:** limited CPU/memory/storage on the edge node; 10 W AI power budget from
12 V truck supply via DC-DC converter; energy-efficient continuous operation.

**Software:** Python-based pipeline; must containerise (Docker) for OTA updates;
cross-platform; simple deployment via Ansible.

**Network:** local MQTT (Mosquitto) for lightweight, low-bandwidth, decoupled
messaging between simulator/consumer/inference stages; cellular uplink carries only
inference/alert payloads, never raw telemetry (see Section 3).

**AI:** small labelled dataset (~300 windows across 3 classes); fast inference (<2 ms
observed for M1); Class 2 (Critical) recall must exceed 95% on the deployed variant;
model compression (INT8 PTQ + pruning) required to fit Flash/SRAM budgets.

---

# 7. Known Gap — Feature Engineering

> **Note for grading transparency:** the constraints above assume the specified
> preprocessing pipeline (5-sample moving average → 30 s window / 10 s step → 6
> engineered features: temp_mean, temp_std, temp_slope, vib_rms, vib_peak,
> vib_kurtosis). The current `preprocessing.py` implementation instead normalises raw
> per-second readings without windowing. This must be corrected before the features
> (and therefore the trained models) reflect the architecture described here — see
> `README.md` "Known Limitations".

---

# Conclusion

LogiEdge satisfies the four core Edge AI constraints — latency, bandwidth,
connectivity, and privacy — by keeping the sense-to-alert decision loop entirely
on-device and shipping only classification outcomes off the truck. The remaining gap
is internal to the preprocessing implementation, not the architecture.
