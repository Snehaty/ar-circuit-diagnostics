import numpy as np


class SummingMixer:
    """
    Digital twin of a 3-channel audio mixer built from a single inverting
    summing amplifier (LM741, dual supply).

    Each channel has:
    - its own input resistor (Rin_n) -> sets that channel's base gain
    - its own volume fader (0.0 to 1.0) -> attenuates the signal before mixing

    All channels sum into one feedback resistor (Rf) through the same
    op-amp inverting input (virtual ground node).

    Vout = -Rf * (fader1*Vin1/Rin1 + fader2*Vin2/Rin2 + fader3*Vin3/Rin3)
    """

    def __init__(self, rin_list=(10_000, 10_000, 10_000), rf=47_000,
                 vcc=9.0, vee=-9.0, gbw=1_000_000, slew_rate=0.5e6):
        self.rin_list = list(rin_list)   # one resistor per channel
        self.rf = rf
        self.vcc = vcc
        self.vee = vee
        self.gbw = gbw
        self.slew_rate = slew_rate
        self.n_channels = len(rin_list)

    def channel_gain(self, ch_index):
        """Gain contribution of a single channel, ignoring fader position."""
        return -self.rf / self.rin_list[ch_index]

    def channel_current(self, ch_index, vin, fader):
        """Current this channel injects into the summing node (amps)."""
        return (fader * vin) / self.rin_list[ch_index]

    def summing_node_current(self, vins, faders):
        """Total current arriving at the virtual ground node from all channels."""
        return sum(self.channel_current(i, vins[i], faders[i])
                   for i in range(self.n_channels))

    def output_voltage_ideal(self, vins, faders):
        """Ideal (unclipped) output voltage."""
        i_total = self.summing_node_current(vins, faders)
        return -self.rf * i_total

    def output_voltage(self, vins, faders):
        """Real output voltage, clipped at supply rails."""
        vout_ideal = self.output_voltage_ideal(vins, faders)
        return np.clip(vout_ideal, self.vee, self.vcc)

    def is_clipped(self, vins, faders):
        vout_ideal = self.output_voltage_ideal(vins, faders)
        return abs(vout_ideal) > max(abs(self.vcc), abs(self.vee))

    def feedback_current(self, vins, faders):
        """Current through Rf -- equals total summing current for ideal op-amp,
        but capped by what the clipped output can actually drive."""
        vout = self.output_voltage(vins, faders)
        return -vout / self.rf

    def per_channel_power(self, vins, faders):
        """Power dissipated in each channel's input resistor (mW)."""
        powers = []
        for i in range(self.n_channels):
            i_ch = self.channel_current(i, vins[i], faders[i])
            p = i_ch**2 * self.rin_list[i] * 1000  # mW
            powers.append(p)
        return powers

    def total_power(self, vins, faders):
        ch_power = sum(self.per_channel_power(vins, faders))
        i_f = self.feedback_current(vins, faders)
        rf_power = i_f**2 * self.rf * 1000
        return ch_power + rf_power

    def summary(self, vins, faders):
        vout = self.output_voltage(vins, faders)
        clipped = self.is_clipped(vins, faders)
        channel_currents_mA = [self.channel_current(i, vins[i], faders[i]) * 1000
                                for i in range(self.n_channels)]
        return {
            "vout": vout,
            "vout_ideal_unclipped": self.output_voltage_ideal(vins, faders),
            "clipped": clipped,
            "channel_currents_mA": channel_currents_mA,
            "feedback_current_mA": self.feedback_current(vins, faders) * 1000,
            "total_power_mW": self.total_power(vins, faders),
        }


if __name__ == "__main__":
    mixer = SummingMixer(rin_list=[10_000, 10_000, 10_000], rf=47_000, vcc=9, vee=-9)

    # Three channels: a vocal, a guitar, a backing track -- example signal levels
    vins = [0.3, 0.2, 0.4]      # volts, peak signal level per channel
    faders = [1.0, 0.5, 0.8]    # 0.0 = muted, 1.0 = full volume

    result = mixer.summary(vins, faders)

    print("Per-channel base gain (Rf/Rin):")
    for i in range(mixer.n_channels):
        print(f"  Channel {i+1}: {mixer.channel_gain(i):.2f}x")

    print()
    print(f"Channel input voltages: {vins}")
    print(f"Fader positions:        {faders}")
    print()
    print(f"Channel currents (mA): {[round(c, 4) for c in result['channel_currents_mA']]}")
    print(f"Output voltage (Vout):  {result['vout']:.3f} V")
    print(f"Clipped:                {result['clipped']}")
    print(f"Total power (mW):       {result['total_power_mW']:.3f}")