import numpy as np

class InvertingAmplifier:
    """
    Digital twin of an inverting op-amp amplifier.
    Includes ideal gain, supply clipping, frequency response, and slew rate limiting.
    """

    def __init__(self, rin=10_000, rf=100_000, vcc=9.0, vee=-9.0,
                 gbw=1_000_000, slew_rate=0.5e6):
        self.rin = rin              # Input resistor (ohms)
        self.rf = rf                # Feedback resistor (ohms)
        self.vcc = vcc              # Positive supply rail
        self.vee = vee              # Negative supply rail
        self.gbw = gbw              # Gain-bandwidth product (Hz) — LM741 ≈ 1MHz
        self.slew_rate = slew_rate  # Slew rate (V/s) — LM741 ≈ 0.5 V/µs = 0.5e6 V/s

    @property
    def gain(self):
        """Ideal DC voltage gain (dimensionless, negative = inverted)"""
        return -self.rf / self.rin

    @property
    def gain_db(self):
        return 20 * np.log10(abs(self.gain))

    @property
    def cutoff_frequency(self):
        """
        Closed-loop bandwidth — the frequency where gain starts rolling off.
        fc = GBW / |gain|
        """
        return self.gbw / abs(self.gain)

    def gain_at_frequency(self, freq):
        """
        Gain magnitude at a given frequency, accounting for GBW rolloff.
        Single-pole model: gain drops -20dB/decade above cutoff frequency.
        """
        ideal_gain = abs(self.gain)
        fc = self.cutoff_frequency
        # Single-pole rolloff formula
        gain_at_f = ideal_gain / np.sqrt(1 + (freq / fc) ** 2)
        return gain_at_f

    def output_voltage(self, vin):
        """DC/low-frequency output, clipped at supply rails."""
        vout_ideal = self.gain * vin
        return np.clip(vout_ideal, self.vee, self.vcc)

    def output_voltage_at_frequency(self, vin_amplitude, freq):
        """Output amplitude at a given frequency, accounting for bandwidth rolloff."""
        gain_f = self.gain_at_frequency(freq)
        vout_ideal = -gain_f * vin_amplitude  # negative sign preserved for inverting
        return np.clip(vout_ideal, self.vee, self.vcc)

    def input_current(self, vin):
        return vin / self.rin

    def feedback_current(self, vin):
        vout = self.output_voltage(vin)
        return -vout / self.rf

    def power_dissipated(self, vin):
        i_in = self.input_current(vin)
        i_f = self.feedback_current(vin)
        return i_in**2 * self.rin + i_f**2 * self.rf

    def is_slew_limited(self, vin_amplitude, freq):
        """
        Check if the required output slew rate exceeds the op-amp's max slew rate.
        For a sine wave Vout = A*sin(2*pi*f*t), max slope = A * 2*pi*f
        """
        vout_amplitude = abs(self.gain) * vin_amplitude
        required_slew_rate = vout_amplitude * 2 * np.pi * freq
        return required_slew_rate > self.slew_rate, required_slew_rate

    def max_undistorted_frequency(self, vin_amplitude):
        """
        The highest frequency at which the output can still be a clean sine wave
        before slew rate distortion kicks in.
        """
        vout_amplitude = abs(self.gain) * vin_amplitude
        if vout_amplitude == 0:
            return float('inf')
        return self.slew_rate / (2 * np.pi * vout_amplitude)

    def frequency_response(self, freq_range):
        """Returns gain in dB across a range of frequencies — for Bode plot."""
        gains = np.array([self.gain_at_frequency(f) for f in freq_range])
        gains_db = 20 * np.log10(np.maximum(gains, 1e-12))
        return gains_db

    def summary(self, vin, freq=0):
        vout = self.output_voltage(vin)
        slew_limited, required_sr = self.is_slew_limited(abs(vin), freq) if freq > 0 else (False, 0)

        return {
            "vin": vin,
            "frequency_Hz": freq,
            "vout": vout,
            "gain": self.gain,
            "gain_db": self.gain_db,
            "cutoff_frequency_Hz": self.cutoff_frequency,
            "gain_at_this_freq": self.gain_at_frequency(freq) if freq > 0 else abs(self.gain),
            "input_current_mA": self.input_current(vin) * 1000,
            "feedback_current_mA": self.feedback_current(vin) * 1000,
            "power_mW": self.power_dissipated(vin) * 1000,
            "clipped": abs(self.gain * vin) > max(abs(self.vcc), abs(self.vee)),
            "slew_rate_limited": slew_limited,
            "required_slew_rate_V_per_s": required_sr,
        }


if __name__ == "__main__":
    amp = InvertingAmplifier(rin=10_000, rf=100_000, vcc=9, vee=-9,
                               gbw=1_000_000, slew_rate=0.5e6)

    print(f"DC Gain: {amp.gain} ({amp.gain_db:.1f} dB)")
    print(f"Cutoff frequency: {amp.cutoff_frequency:.0f} Hz")
    print()

    # Test gain rolloff at different frequencies
    test_freqs = [100, 1_000, 10_000, 100_000]
    for f in test_freqs:
        g = amp.gain_at_frequency(f)
        print(f"At {f} Hz: gain = {g:.2f}x ({20*np.log10(g):.1f} dB)")

    print()
    print(f"Max undistorted frequency at Vin=0.5V: {amp.max_undistorted_frequency(0.5):.0f} Hz")