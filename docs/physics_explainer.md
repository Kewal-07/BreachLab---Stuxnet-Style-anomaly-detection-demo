# The physics, in plain English

> A cheat-sheet for explaining BreachLab's simulator without equations. Safe to
> lift straight into a LinkedIn post, README, or interview answer.

## The one sentence that matters

Every signal in the simulation is derived from **one true rotor speed**. Change
the speed and everything else follows automatically. That's why the streams can
never contradict each other — and why an attack that damages the machine
*cannot* hide from the physical signals, even while it lies to the operator's
screen.

> ⚠️ The numbers are **illustrative, not calibrated to real hardware.** The
> point is a plausible, internally-consistent signal — not an engineering model
> of a real centrifuge. Say this out loud; it's the honest and normal caveat.

## The five signals (four sentences of intuition)

| Signal | Plain-English story | Why it matters for the demo |
|--------|--------------------|-----------------------------|
| **rotor_speed** | How fast the centrifuge spins. | This is the thing the attack secretly manipulates. |
| **vibration** | The further speed drifts from normal, the more it shakes — and it grows *fast* (quadratically), not gently. | The physical tell-tale. An over-speed spike lights this up immediately. |
| **motor_power** | Roughly tracks how hard the motor is working. | A second independent witness that moves with speed. |
| **casing_temp** | Heats up as the motor works — but **slowly**. It lags; it can't jump instantly. | The lag is the domain-knowledge gem: a faked "everything's fine" reading can't reproduce realistic thermal inertia. |
| **bearing_wear** | Creeps upward over the whole run and never goes back down. | Slow background drift — makes the baseline realistically imperfect. |

## Why this beats "just watch one number"

Because the signals are *coupled*, the strongest thing you can say is:

> "The detector doesn't flag a single threshold being crossed. It flags that the
> signals **stopped agreeing with each other** — speed spiked but the operator's
> screen stayed calm, and vibration and temperature told a different story than
> the reported speed. That disagreement is the fingerprint of the attack."

## The "loop of fiction" (the Stuxnet hook)

Real Stuxnet recorded ~21 seconds of normal readings and **replayed them to
operators** while it drove the centrifuges to destruction. Operators watched a
calm, normal screen during the sabotage. BreachLab reproduces exactly this: the
`hmi_reported` stream gets a looped recording of normal telemetry, while the
`physical` stream shows the real, damaging behaviour.

**The finding:** run the same anomaly detector on both streams. On the operator
(HMI) stream it sees nothing — that's the fiction working. On the physical
stream it catches the attack. Same detector, opposite result. That's the post.

## Benign anomalies (why the baseline isn't suspiciously clean)

Real plants aren't perfectly smooth, so the simulation includes non-malicious
oddities the detector must learn to *ignore*:

- **Maintenance dips** — brief planned slow-downs (like an operator throttling a
  unit).
- **Sensor glitches** — a single flaky reading from a sensor.
- **Bearing wear** — the slow drift above.

If a detector cried "attack!" at these, it would be useless in the real world.
Tolerating them is what makes the results honest.
