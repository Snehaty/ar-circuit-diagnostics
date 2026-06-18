import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from opamp_model import InvertingAmplifier

st.set_page_config(page_title="Op-Amp Digital Twin", page_icon="⚡", layout="wide")

st.title("⚡ Inverting Amplifier — Digital Twin")
st.caption("Virtual control panel for an op-amp based inverting amplifier")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Circuit Parameters")
    rin = st.slider("Rin (kΩ)", 1, 100, 10) * 1000
    rf = st.slider("Rf (kΩ)", 1, 500, 100) * 1000
    vcc = st.slider("Supply +Vcc (V)", 3.0, 15.0, 9.0, 0.5)
    vee = -vcc
    vin = st.slider("Input voltage Vin (V)", -3.0, 3.0, 0.5, 0.1)

    st.divider()
    st.subheader("Op-amp Non-idealities")
    opamp_choice = st.selectbox("Op-amp model", ["LM741 (GBW=1MHz, SR=0.5V/µs)",
                                                    "TL072 (GBW=3MHz, SR=13V/µs)",
                                                    "Custom"])
    if opamp_choice.startswith("LM741"):
        gbw, slew_rate = 1_000_000, 0.5e6
    elif opamp_choice.startswith("TL072"):
        gbw, slew_rate = 3_000_000, 13e6
    else:
        gbw = st.number_input("GBW (Hz)", value=1_000_000, step=100_000)
        slew_rate = st.number_input("Slew rate (V/s)", value=0.5e6, step=0.1e6) 

    signal_freq = st.slider("Signal frequency (Hz)", 10, 200_000, 1000, 10)

    amp = InvertingAmplifier(rin=rin, rf=rf, vcc=vcc, vee=vee, gbw=gbw, slew_rate=slew_rate)
    result = amp.summary(vin, freq=signal_freq)

    st.divider()
    st.subheader("Live Output")

    if result["clipped"]:
        st.error(" Output clipped! Op-amp saturated at supply rail.")
    if result["slew_rate_limited"]:
        st.warning(f" Slew rate limited! Output will distort into a triangle wave at this frequency.")

    st.metric("Output voltage (Vout, DC)", f"{result['vout']:.3f} V")
    st.metric("Gain at this frequency", f"{result['gain_at_this_freq']:.2f}x")
    st.metric("Cutoff frequency (-3dB)", f"{result['cutoff_frequency_Hz']:,.0f} Hz")
    st.metric("Max distortion-free freq", f"{amp.max_undistorted_frequency(abs(vin)):,.0f} Hz")

with col2:
    st.subheader("Bode Plot — Gain vs Frequency")

    freqs = np.logspace(1, 6, 300)  # 10 Hz to 1 MHz
    gains_db = amp.frequency_response(freqs)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.semilogx(freqs, gains_db, color="#378ADD", linewidth=2)
    ax1.axhline(amp.gain_db - 3, color="red", linestyle="--", alpha=0.6, label="-3dB point")
    ax1.axvline(amp.cutoff_frequency, color="red", linestyle="--", alpha=0.4)
    ax1.axvline(signal_freq, color="green", linestyle="-", alpha=0.5, label="Current signal freq")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Gain (dB)")
    ax1.set_title("Frequency Response")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")
    st.pyplot(fig1)

    st.subheader("Output waveform at this frequency")

    t = np.linspace(0, 3 / signal_freq, 500)  # show 3 cycles
    vin_wave = vin * np.sin(2 * np.pi * signal_freq * t)

    # Apply gain rolloff
    gain_f = amp.gain_at_frequency(signal_freq)
    vout_ideal = -gain_f * vin_wave

    # Apply slew rate limiting (simple model: clip the rate of change)
    dt = t[1] - t[0]
    max_step = amp.slew_rate * dt
    vout_slewed = np.copy(vout_ideal)
    for i in range(1, len(vout_slewed)):
        delta = vout_slewed[i] - vout_slewed[i-1]
        if abs(delta) > max_step:
            vout_slewed[i] = vout_slewed[i-1] + np.sign(delta) * max_step

    vout_final = np.clip(vout_slewed, vee, vcc)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(t * 1000, vin_wave, label="Vin", color="#378ADD")
    ax2.plot(t * 1000, vout_final, label="Vout (real)", color="#D85A30")
    ax2.axhline(vcc, color="gray", linestyle="--", alpha=0.4)
    ax2.axhline(vee, color="gray", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Voltage (V)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    if result["slew_rate_limited"]:
        st.caption(" Notice the output looks more like a triangle wave than a sine — that's slew rate distortion, not clipping.")