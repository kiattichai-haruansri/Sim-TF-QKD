# main.py
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

    print("==========================")
    print("Starting TF-QKD Simulation")
    print("==========================")
    print(f"Distance : {config.FIBER_LENGTH_KM*2} km")
    print(f"Total Pulses : {total_pulses:,}")
    print(f"Batch Size   : {batch_size:,}")

    alice = Alice()
    bob = Bob()
    charlie = Charlie()

    fiber_a = Fiber(config.FIBER_LENGTH_KM, config.FIBER_LOSS_DB_PER_KM)
    fiber_b = Fiber(config.FIBER_LENGTH_KM, config.FIBER_LOSS_DB_PER_KM)

    pair_counts = {}

    for muA in config.INTENSITIES:
        for muB in config.INTENSITIES:
            pair_counts[(muA, muB)] = {
                "sent": 0,
                "valid": 0,
                "error": 0,
            }

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

        valid = phase_match & (d0 ^ d1)

        error = valid & (
            ((rec_a["encoding_phase"] == rec_b["encoding_phase"]) & d1)
            |
            ((rec_a["encoding_phase"] != rec_b["encoding_phase"]) & d0)
        )

        for muA in config.INTENSITIES:
            for muB in config.INTENSITIES:

                mask = (
                    (rec_a["intensity"] == muA)
                    &
                    (rec_b["intensity"] == muB)
                )

                pair_counts[(muA,muB)]["sent"] += int(np.sum(mask))
                pair_counts[(muA,muB)]["valid"] += int(np.sum(valid & mask))
                pair_counts[(muA,muB)]["error"] += int(np.sum(error & mask))

    """
    def build(mu):
        total = valid_clicks[mu]
        err = error_clicks[mu]

        return {
            "gain": total / sent_counts[mu] if sent_counts[mu] else 0.0,
            "qber": err / total if total else 0.0,
            "total_events": total,
        }
    """
    
    stats = {}

    for pair,data in pair_counts.items():

        sent = data["sent"]
        valid = data["valid"]
        err = data["error"]

        stats[pair] = {
            "gain": valid/sent if sent else 0,
            "qber": err/valid if valid else 0,
            "events": valid
        }
    """
    sig = build(config.MU_SIGNAL)
    dec = build(config.MU_DECOY)
    vac = build(config.MU_VACUUM)
    """

    labels = {
        config.MU_SIGNAL: "Signal",
        config.MU_DECOY: "Decoy",
        config.MU_VACUUM: "Vacuum",
    }

    print("\n--- 2. Data Sifting & QBER Analysis ---")

    for (muA, muB), s in stats.items():

        nameA = labels.get(muA, str(muA))
        nameB = labels.get(muB, str(muB))

        print(
            f"[{nameA}-{nameB}] "
            f"(μA={muA}, μB={muB}) | "
            f"Gain: {s['gain']:.2e} | "
            f"QBER: {s['qber']*100:.2f}% | "
            f"Clicks: {s['events']:,}"
        )

    print("========================================")
    print(f"Simulation Time: {time.time()-start:.2f} s")

if __name__ == "__main__":
    run_vectorized_tfqkd(
        total_pulses=10**10,
        batch_size=10**6,
    )
