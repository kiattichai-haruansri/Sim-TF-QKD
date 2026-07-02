# main.py
"""
Template Vectorized TF-QKD main.

NOTE:
This file assumes the batch APIs discussed during refactoring:
- Alice.prepare_batch(...)
- Bob.prepare_batch(...)
- Fiber.propagate_batch(...)
- Charlie.measure_batch(...) -> (d0_click, d1_click)

Adjust names if your final component APIs differ.
"""

import time
import numpy as np
from tqdm import tqdm

import config

from protocol.alice import Alice
from protocol.bob import Bob
from protocol.charlie import Charlie
from components.fiber import Fiber

from analysis.parameter_estimation import estimate_decoy_parameters
from analysis.key_rate import calculate_finite_key_rate


def run_vectorized_tfqkd(total_pulses=10**8, batch_size=10**6):

    print("========================================")
    print(" 🚀 Starting Vectorized TF-QKD Simulation 🚀")
    print("========================================")
    print(f"Distance : {config.FIBER_LENGTH_KM*2} km")
    print(f"Total Pulses : {total_pulses:,}")
    print(f"Batch Size   : {batch_size:,}")

    alice = Alice()
    bob = Bob()
    charlie = Charlie()

    fiber_a = Fiber(config.FIBER_LENGTH_KM, config.FIBER_LOSS_DB_PER_KM)
    fiber_b = Fiber(config.FIBER_LENGTH_KM, config.FIBER_LOSS_DB_PER_KM)

    sent_counts = {mu:0 for mu in config.INTENSITIES}
    valid_clicks = {mu:0 for mu in config.INTENSITIES}
    error_clicks = {mu:0 for mu in config.INTENSITIES}

    phase_tol = 0.15

    batches = total_pulses // batch_size

    start = time.time()
    print("\n--- 1. Physical Layer Simulation ---")

    for _ in tqdm(range(batches), desc="Simulation"):

        pulse_a, rec_a = alice.prepare_batch(
            batch_size,
            config.INTENSITIES,
            config.ENCODING_PHASES,
        )

        pulse_b, rec_b = bob.prepare_batch(
            batch_size,
            config.INTENSITIES,
            config.ENCODING_PHASES,
        )

        pulse_a = fiber_a.propagate_batch(pulse_a)
        pulse_b = fiber_b.propagate_batch(pulse_b)

        result = charlie.measure_batch(
            pulse_a,
            pulse_b,
        )

        d0 = result["D0"]
        d1 = result["D1"]

        same_mu = rec_a["intensity"] == rec_b["intensity"]

        phase_diff = np.abs(
            rec_a["global_phase"] - rec_b["global_phase"]
        ) % (2*np.pi)

        phase_diff = np.minimum(
            phase_diff,
            2*np.pi - phase_diff
        )

        phase_match = phase_diff <= phase_tol

        valid = same_mu & phase_match & (d0 ^ d1)

        error = valid & (
            ((rec_a["encoding_phase"] == rec_b["encoding_phase"]) & d1)
            |
            ((rec_a["encoding_phase"] != rec_b["encoding_phase"]) & d0)
        )

        for mu in config.INTENSITIES:

            mask = rec_a["intensity"] == mu

            sent_counts[mu] += int(np.sum(same_mu & mask))
            valid_clicks[mu] += int(np.sum(valid & mask))
            error_clicks[mu] += int(np.sum(error & mask))

    def build(mu):
        total = valid_clicks[mu]
        err = error_clicks[mu]

        return {
            "gain": total / sent_counts[mu] if sent_counts[mu] else 0.0,
            "qber": err / total if total else 0.0,
            "total_events": total,
        }

    sig = build(config.MU_SIGNAL)
    dec = build(config.MU_DECOY)
    vac = build(config.MU_VACUUM)

    print("\n--- 2. Data Sifting & QBER Analysis ---")

    print(
        f"[Signal (mu={config.MU_SIGNAL})] "
        f"Gain: {sig['gain']:.2e} | "
        f"QBER: {sig['qber']*100:.2f}% | "
        f"Clicks: {sig['total_events']:,}"
    )

    print(
        f"[Decoy  (nu={config.MU_DECOY})] "
        f"Gain: {dec['gain']:.2e} | "
        f"QBER: {dec['qber']*100:.2f}% | "
        f"Clicks: {dec['total_events']:,}"
    )

    print(
        f"[Vacuum (0={config.MU_VACUUM})] "
        f"Gain: {vac['gain']:.2e} | "
        f"QBER: {vac['qber']*100:.2f}% | "
        f"Clicks: {vac['total_events']:,}"
    )

    print("\n--- 3. Parameter Estimation & Key Rate ---")

    params = estimate_decoy_parameters(
        mu=config.MU_SIGNAL,
        nu=config.MU_DECOY,
        gain_mu=sig["gain"],
        gain_nu=dec["gain"],
        gain_vac=vac["gain"],
        error_mu=sig["qber"],
        error_nu=dec["qber"],
    )

    print(f"Estimated Single-photon Yield (Y₁): {params['Y_1']:.2e}")
    print(f"Estimated Single-photon Error (e₁): {params['e_1']*100:.2f}%")

    skr = calculate_finite_key_rate(
        M_x=sig["total_events"],
        qber_signal=sig["qber"],
        e_ph_U=params["e_1"],
        epsilon_s=1e-10,
        lambda_ec=config.LAMBDA_EC,
    )

    print("\n========================================")

    if skr > 0:

        total_secure_bits = int(
            skr * sent_counts[config.MU_SIGNAL]
        )

        print(f"✅ SECURE KEY RATE (Finite-Key): {skr:.6e} bits/pulse")
        print(f"✅ Total Secure Bits Extracted: ~{total_secure_bits:,} bits")

    else:

        print("❌ SECURE KEY RATE: 0.0 bits/pulse")
        print("❌ Key rate is zero due to high finite-key penalties or high noise.")

    print("========================================")
    print(f"Simulation Time: {time.time()-start:.2f} s")

if __name__ == "__main__":
    run_vectorized_tfqkd(
        total_pulses=10**10,
        batch_size=10**6,
    )
