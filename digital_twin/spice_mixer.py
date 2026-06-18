import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()


def build_summing_mixer_circuit(rin_values, rf_value, vin_values, vcc=9, vee=-9):
    """
    Builds a real SPICE circuit of a 3-channel summing inverting amplifier.
    Uses an ideal high-gain op-amp behavioral model (standard SPICE practice
    when a vendor subcircuit isn't loaded) -- this still solves real
    Kirchhoff's current law equations via ngspice, not hand-derived formulas.
    """
    circuit = Circuit("3-Channel Summing Mixer")

    circuit.V("CC", "vcc", circuit.gnd, vcc @ u_V)
    circuit.V("EE", "vee", circuit.gnd, vee @ u_V)

    for i, vin in enumerate(vin_values):
        circuit.V(f"in{i+1}", f"vin{i+1}", circuit.gnd, vin @ u_V)

    for i, rin in enumerate(rin_values):
        circuit.R(f"in{i+1}", f"vin{i+1}", "vsum", rin @ u_Ohm)

    circuit.R("f", "vout", "vsum", rf_value @ u_Ohm)

    circuit.VCVS("opamp", "vout", circuit.gnd, circuit.gnd, "vsum", voltage_gain=100000)

    return circuit


def simulate_mixer(rin_values, rf_value, vin_values, vcc=9, vee=-9):
    """Runs the SPICE DC operating point analysis and returns key results."""
    circuit = build_summing_mixer_circuit(rin_values, rf_value, vin_values, vcc, vee)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.operating_point()

    # analysis["vout"] is a numpy array, not a plain scalar -- index into it first
    vout = float(analysis["vout"][0])
    vsum = float(analysis["vsum"][0])

    channel_currents_mA = []
    for i, (rin, vin) in enumerate(zip(rin_values, vin_values)):
        i_ch = (vin - vsum) / rin
        channel_currents_mA.append(i_ch * 1000)

    feedback_current_mA = (vout - vsum) / rf_value * 1000

    return {
        "vout": vout,
        "vsum_virtual_ground": vsum,
        "channel_currents_mA": channel_currents_mA,
        "feedback_current_mA": feedback_current_mA,
        "clipped": abs(vout) >= max(abs(vcc), abs(vee)) - 0.1,
    }


if __name__ == "__main__":
    rin_values = [10_000, 10_000, 10_000]
    rf_value = 47_000
    vin_values = [0.3 * 1.0, 0.2 * 0.5, 0.4 * 0.8]  # vin * fader, precomputed

    result = simulate_mixer(rin_values, rf_value, vin_values)

    print("SPICE-solved results:")
    print(f"  Vout:                  {result['vout']:.4f} V")
    print(f"  Virtual ground (vsum): {result['vsum_virtual_ground']:.6f} V  (should be ~0)")
    print(f"  Channel currents (mA): {[round(c, 4) for c in result['channel_currents_mA']]}")
    print(f"  Feedback current (mA): {result['feedback_current_mA']:.4f}")
    print(f"  Clipped:               {result['clipped']}")