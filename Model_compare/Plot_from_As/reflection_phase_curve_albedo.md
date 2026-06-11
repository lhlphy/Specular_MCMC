# From Reflection Phase Curve to Geometric, Spherical, and Bond Albedo

This note summarizes how to estimate **geometric albedo** $A_g$, **spherical albedo** $A_s$, and **Bond albedo** $A_B$ from an observed or modeled **reflection phase curve** of an exoplanet.

We assume the thermal emission component has already been removed from the phase curve.

---

## 1. Input Quantity

Let the reflection phase curve be

$$
F(\theta)=\frac{F_p}{F_*},
$$

where:

- $F_p$ is the reflected planetary flux;
- $F_*$ is the stellar flux;
- $F(\theta)$ is often given in **ppm**;
- $\theta=0$ corresponds to **transit**, where the observer sees the planetary nightside;
- $\theta=\pi$ corresponds to **secondary eclipse**, where the observer would see the fully illuminated dayside just before or after eclipse.

If $F(\theta)$ is given in ppm, convert it to a dimensionless flux ratio:

$$
f(\theta)=F(\theta)\times 10^{-6}.
$$

---

## 2. Required Parameters

To infer albedo, one needs the orbital distance $a$ or an equivalent normalized separation.

The standard reflected-light phase curve expression is

$$
\frac{F_p}{F_*}
=
A_g
\left(\frac{R_p}{a}\right)^2
\Phi(\alpha),
$$

where:

- $A_g$ is the geometric albedo;
- $R_p$ is the planetary radius;
- $a$ is the star--planet separation;
- $\Phi(\alpha)$ is the normalized phase function;
- $\Phi(0)=1$;
- $\alpha$ is the phase angle, with $\alpha=0$ at full phase and $\alpha=\pi$ at new phase.

For a circular, nearly edge-on orbit with the convention used here,

$$
\alpha = |\pi-\theta|.
$$

For $0\le \theta \le \pi$,

$$
\alpha = \pi-\theta.
$$

If one has $a/R_*$ and $R_p/R_*$ from transit fitting, then

$$
\frac{a}{R_p}
=
\frac{a/R_*}{R_p/R_*}.
$$

> **Important:** Knowing only $R_p$ and $R_*$ is generally not enough. One also needs $a$, $a/R_*$, or an equivalent orbital-distance constraint.

---

## 3. Geometric Albedo $A_g$

At full phase, $\alpha=0$ and $\Phi(0)=1$. This corresponds to $\theta=\pi$.

Therefore,

$$
f(\pi)
=
A_g
\left(\frac{R_p}{a}\right)^2.
$$

Thus,

$$
\boxed{
A_g
=
f(\pi)
\left(\frac{a}{R_p}\right)^2
}
$$

or, if the phase curve is in ppm,

$$
\boxed{
A_g
=
F(\pi)\times 10^{-6}
\left(\frac{a}{R_p}\right)^2
}.
$$

In practice, the planet is hidden during secondary eclipse, so $F(\pi)$ should usually be obtained by fitting or interpolating/extrapolating the reflection phase curve immediately before and after eclipse, rather than by directly reading the in-eclipse flux.

---

## 4. Phase Function and Phase Integral

The normalized phase function is

$$
\Phi(\alpha)
=
\frac{f(\theta)}{f(\pi)},
$$

where

$$
\alpha=|\pi-\theta|.
$$

The phase integral is defined as

$$
q
=
2\int_0^\pi \Phi(\alpha)\sin\alpha\,d\alpha.
$$

The spherical albedo is related to the geometric albedo through

$$
\boxed{
A_s=qA_g
}.
$$

---

## 5. Spherical Albedo $A_s$

The spherical albedo at a given wavelength or within a given observing band is the fraction of incident radiation reflected into all directions.

Using the reflection phase curve directly, one may write

$$
\boxed{
A_s
=
2
\left(\frac{a}{R_p}\right)^2
\int_0^\pi f(\theta)\sin\theta\,d\theta
}.
$$

If $F(\theta)$ is in ppm,

$$
\boxed{
A_s
=
2
\left(\frac{a}{R_p}\right)^2
\int_0^\pi
F(\theta)\times 10^{-6}
\sin\theta\,d\theta
}.
$$

This form follows from $\alpha=\pi-\theta$ and $\sin\alpha=\sin\theta$.

---

## 6. Full-Orbit Phase Curve

If a full-orbit phase curve from $0$ to $2\pi$ is available, and the brightness is not perfectly symmetric before and after eclipse, first define a phase-angle-averaged flux ratio:

$$
\bar f(\alpha)
=
\frac{f(\pi-\alpha)+f(\pi+\alpha)}{2}.
$$

Then

$$
\boxed{
A_s
=
2
\left(\frac{a}{R_p}\right)^2
\int_0^\pi \bar f(\alpha)\sin\alpha\,d\alpha
}.
$$

Equivalently, one can compute

$$
\Phi(\alpha)=\frac{\bar f(\alpha)}{\bar f(0)}
$$

and then use

$$
q=2\int_0^\pi \Phi(\alpha)\sin\alpha\,d\alpha,
$$

$$
A_s=qA_g.
$$

---

## 7. Bond Albedo $A_B$

The Bond albedo is the total fraction of incident stellar radiation reflected by the planet, integrated over all wavelengths and all outgoing directions.

It is the stellar-spectrum-weighted spherical albedo:

$$
\boxed{
A_B
=
\frac{
\int A_s(\lambda)F_*(\lambda)\,d\lambda
}{
\int F_*(\lambda)\,d\lambda
}
}.
$$

Therefore, a phase curve measured in a single photometric band gives only a **band-integrated spherical albedo**, not the true Bond albedo.

For a specific observing band $T(\lambda)$, the measured quantity is closer to

$$
A_s^{\rm band}
=
\frac{
\int A_s(\lambda)F_*(\lambda)T(\lambda)\,d\lambda
}{
\int F_*(\lambda)T(\lambda)\,d\lambda
}.
$$

A crude approximation sometimes used is

$$
A_B \approx A_s^{\rm band},
$$

but this is only reasonable if:

1. the observing band covers most of the stellar irradiation energy, and
2. the planetary reflectivity does not vary strongly with wavelength.

For most real exoplanets, especially hot planets with strong wavelength-dependent scattering and absorption, this approximation can be poor.

---

## 8. Discrete Numerical Evaluation

Suppose one has measurements $F_i=F(\theta_i)$ in ppm for $0\le \theta_i\le\pi$.

Convert to dimensionless flux ratios:

$$
f_i=F_i\times 10^{-6}.
$$

Then estimate geometric albedo as

$$
A_g
\approx
f(\pi)
\left(\frac{a}{R_p}\right)^2.
$$

For spherical albedo, use numerical quadrature:

$$
A_s
\approx
2
\left(\frac{a}{R_p}\right)^2
\sum_i f_i\sin\theta_i\Delta\theta_i.
$$

More robustly, use trapezoidal integration:

$$
A_s
\approx
2
\left(\frac{a}{R_p}\right)^2
\operatorname{trapz}\left[f(\theta)\,\sin\theta,\theta\right].
$$

In Python-like notation:

```python
import numpy as np

# theta: orbital phase angle in radians, from 0 to pi
# F_ppm: reflected-light phase curve in ppm
# a_over_Rp: a / Rp

f = F_ppm * 1e-6

Ag = f_at_eclipse * a_over_Rp**2
As = 2 * a_over_Rp**2 * np.trapz(f * np.sin(theta), theta)
```

Here `f_at_eclipse` should represent the modeled or extrapolated full-phase value $f(\pi)$, not the in-eclipse flux.

---

## 9. Summary of Main Formulae

Given a reflected-light phase curve $F(\theta)$ in ppm:

### Geometric albedo

$$
\boxed{
A_g
=
F(\pi)\times 10^{-6}
\left(\frac{a}{R_p}\right)^2
}
$$

### Spherical albedo

$$
\boxed{
A_s
=
2
\left(\frac{a}{R_p}\right)^2
\int_0^\pi
F(\theta)\times 10^{-6}
\sin\theta\,d\theta
}
$$

### Bond albedo

$$
\boxed{
A_B
=
\frac{
\int A_s(\lambda)F_*(\lambda)\,d\lambda
}{
\int F_*(\lambda)\,d\lambda
}
}
$$

---

## 10. Practical Notes

1. **Thermal emission must be removed** before interpreting the phase curve as reflected light.
2. **Secondary eclipse points should not be used directly** for $F(\pi)$, because the planet is occulted.
3. **The orbital distance $a$ is required.** If only $R_p$ and $R_*$ are known, the albedo cannot be determined uniquely.
4. **Single-band observations do not directly give Bond albedo.** They give bandpass-dependent geometric and spherical albedos.
5. **Phase-curve asymmetry matters.** East--west brightness offsets can change the inferred phase integral. If the full orbit is available, use phase-angle averaging or a physically motivated phase-function model.
6. **Negative flux values after detrending are unphysical for pure reflected light.** They usually indicate noise, imperfect thermal subtraction, systematics, or an overfitted baseline.
