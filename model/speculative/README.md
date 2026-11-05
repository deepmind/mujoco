# Speculative contacts

Example models demonstrating continuous collision detection via speculative contacts
(`<flag speculative="enable"/>`), which prevents fast-moving or thin rigid bodies from tunneling
through obstacles within a single timestep. See the
[Speculative contacts documentation](https://mujoco.readthedocs.io/en/latest/computation/speculative.html)
for the underlying formulation.

| Model                        | Timestep | Description                                            |
| ---------------------------- | -------- | ------------------------------------------------------ |
| [`newton.xml`](newton.xml)   | 4 ms     | Five 3 mm steel plates suspended on twin wires         |
| [`chain.xml`](chain.xml)     | 6 ms     | Free-falling chain of thin rectangular capsule links   |
| [`flipper.xml`](flipper.xml) | 10 ms    | Spring-launched ball struck by a high-speed hinged bat |

Disable the `speculative` flag in any of these models (`<flag speculative="disable"/>` or in the
`simulate` UI) to observe the geoms pass through each other at the nominal timestep.

## Models

* **Newton's cradle ([`newton.xml`](newton.xml)):** Five 3 mm thick steel plates
  (`size="0.0015 0.08 0.08"`) hang from twin wires. The first plate is released from 58° and
  reaches ~3 m/s (12 mm per 4 ms step, or 4× the plate thickness) at the bottom of its swing.
  Without speculative contacts, the swinging plate passes straight through its neighbors.
* **Interlocking chain ([`chain.xml`](chain.xml)):** Four free-jointed rectangular links built from
  6 mm radius capsules drop 28 cm onto a fixed horizontal catch bar at a 6 ms timestep. Speculative
  contacts catch both the top link on the bar and the thin capsule bars of each link against the
  next as the shock wave travels down the chain.
* **Baseball and flipper ([`flipper.xml`](flipper.xml)):** A spring-loaded plunger launches a 4 cm
  ball toward a 1.2 cm thick hinged bat that whips through the impact zone under a stiff torsional
  spring at a 10 ms timestep. Both the ball's linear travel and the bat's rotational sweep exceed
  their thicknesses during the impact step.

## Modeling notes

* **Elastic restitution ([`newton.xml`](newton.xml)):** Standard speculative contacts split impact
  across two steps—arresting `(1 - t)` of the normal speed to land on the surface on the first
  step, and resolving surface contact on the next. When modeling nearly elastic collisions via
  direct negative `solref` (so `h * damping ≈ 2` reverses the normal velocity in a single step),
  set the stiffness to zero (`solref="0 -499"`) with a stiff impedance
  (`solimp="0.995 0.995 0.001"`) so the speculative position spring term does not cancel out the
  elastic damper during the approach step (noting that zero stiffness provides no static position
  spring for resting contacts).
* **Single-pass pre-step velocities:** Because speculative sweeps use pre-step velocities
  (`mjData.qvel`), a stationary intermediate body struck by a fast neighbor acquires its
  post-impact velocity only after the solver runs. In multi-body impact chains (such as
  [`newton.xml`](newton.xml)), keep a small clearance between adjacent plates or choose a timestep
  where a single step's travel does not jump across two gaps at once.

## Changelog

* 27-10-2026: Initial release.
