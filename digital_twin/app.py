import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from opamp_model import SummingMixer

st.set_page_config(page_title="3-Channel Mixer — Digital Twin", page_icon=None, layout="wide")


def draw_circuit_svg(rin_list, rf, faders):
    """Returns an SVG string of the 3-channel summing mixer schematic."""
    fader_labels = [f"{f:.1f}" for f in faders]
    rin_labels = [f"{r/1000:.0f}k" for r in rin_list]

    svg = f"""
    <svg width="100%" height="320" viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg">
      <style>
        .wire {{ stroke: #888; stroke-width: 2; fill: none; }}
        .label {{ font-family: sans-serif; font-size: 13px; fill: #ccc; }}
        .small {{ font-family: sans-serif; font-size: 11px; fill: #888; }}
        .resistor {{ stroke: #4a9eff; stroke-width: 2; fill: none; }}
        .opamp {{ stroke: #f0a030; stroke-width: 2; fill: #2a2a2a; }}
      </style>

      <line x1="20" y1="60" x2="100" y2="60" class="wire"/>
      <path d="M100,60 h10 l5,-8 l10,16 l10,-16 l10,16 l10,-16 l5,8 h10" class="resistor"/>
      <line x1="160" y1="60" x2="260" y2="60" class="wire"/>
      <text x="20" y="48" class="label">Vin1 (fader {fader_labels[0]})</text>
      <text x="105" y="48" class="small">Rin1={rin_labels[0]}</text>

      <line x1="20" y1="130" x2="100" y2="130" class="wire"/>
      <path d="M100,130 h10 l5,-8 l10,16 l10,-16 l10,16 l10,-16 l5,8 h10" class="resistor"/>
      <line x1="160" y1="130" x2="260" y2="130" class="wire"/>
      <line x1="260" y1="60" x2="260" y2="130" class="wire"/>
      <text x="20" y="118" class="label">Vin2 (fader {fader_labels[1]})</text>
      <text x="105" y="118" class="small">Rin2={rin_labels[1]}</text>

      <line x1="20" y1="200" x2="100" y2="200" class="wire"/>
      <path d="M100,200 h10 l5,-8 l10,16 l10,-16 l10,16 l10,-16 l5,8 h10" class="resistor"/>
      <line x1="160" y1="200" x2="260" y2="200" class="wire"/>
      <line x1="260" y1="130" x2="260" y2="200" class="wire"/>
      <text x="20" y="188" class="label">Vin3 (fader {fader_labels[2]})</text>
      <text x="105" y="188" class="small">Rin3={rin_labels[2]}</text>

      <line x1="260" y1="130" x2="320" y2="130" class="wire"/>
      <circle cx="260" cy="130" r="3" fill="#888"/>
      <text x="180" y="148" class="small">summing node</text>

      <polygon points="320,90 320,170 410,130" class="opamp"/>
      <text x="335" y="115" class="label">-</text>
      <text x="335" y="155" class="label">+</text>
      <text x="345" y="135" class="small">LM741</text>

      <line x1="300" y1="160" x2="320" y2="160" class="wire"/>
      <line x1="300" y1="160" x2="300" y2="180" class="wire"/>
      <line x1="290" y1="180" x2="310" y2="180" class="wire"/>
      <line x1="293" y1="186" x2="307" y2="186" class="wire"/>
      <line x1="296" y1="192" x2="304" y2="192" class="wire"/>
      <text x="280" y="205" class="small">GND</text>

      <!-- Vout: extended horizontal run before the corner -->
      <line x1="410" y1="130" x2="480" y2="130" class="wire"/>
      <circle cx="480" cy="130" r="3" fill="#888"/>
      <text x="488" y="118" class="label">Vout</text>

      <!-- Feedback path: vertical up from Vout node, then through Rf zigzag, then down to summing node -->
      <line x1="480" y1="130" x2="480" y2="60" class="wire"/>
      <line x1="480" y1="60" x2="430" y2="60" class="wire"/>
      <path d="M430,60 h-10 l-5,-8 l-10,16 l-10,-16 l-10,16 l-10,-16 l-5,8 h-10" class="resistor"/>
      <line x1="360" y1="60" x2="260" y2="60" class="wire"/>
      <text x="345" y="48" class="small">Rf={rf/1000:.0f}k</text>

      <text x="345" y="85" class="small">+Vcc</text>
      <text x="345" y="180" class="small">-Vee</text>
    </svg>
    """
    return svg

st.title("3-Channel Audio Mixer — Digital Twin")
st.caption("Inverting summing amplifier built around a single LM741, dual supply")

left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Circuit parameters")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        rin1 = st.number_input("Rin 1 (kOhm)", 1, 100, 10) * 1000
    with col_b:
        rin2 = st.number_input("Rin 2 (kOhm)", 1, 100, 10) * 1000
    with col_c:
        rin3 = st.number_input("Rin 3 (kOhm)", 1, 100, 10) * 1000

    rf = st.slider("Feedback resistor Rf (kOhm)", 1, 200, 47) * 1000
    vcc = st.slider("Supply +Vcc (V)", 3.0, 15.0, 9.0, 0.5)
    vee = -vcc

    mixer = SummingMixer(rin_list=[rin1, rin2, rin3], rf=rf, vcc=vcc, vee=vee)

    st.divider()
    st.subheader("Mixing console")

    channel_cols = st.columns(3)
    vins = []
    faders = []
    channel_names = ["Channel 1 (Vocal)", "Channel 2 (Guitar)", "Channel 3 (Backing track)"]

    for i, col in enumerate(channel_cols):
        with col:
            st.markdown(f"**{channel_names[i]}**")
            vin = st.slider("Input level (V)", 0.0, 1.0, 0.3, 0.05, key=f"vin_{i}")
            fader = st.slider("Fader", 0.0, 1.0, 0.8, 0.05, key=f"fader_{i}")
            vins.append(vin)
            faders.append(fader)

    result = mixer.summary(vins, faders)

    st.divider()
    st.subheader("Master output")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Output voltage (Vout)", f"{result['vout']:.3f} V")
    with m2:
        st.metric("Feedback current", f"{result['feedback_current_mA']:.3f} mA")
    with m3:
        st.metric("Total power", f"{result['total_power_mW']:.3f} mW")

    if result["clipped"]:
        st.error(f"Output clipped at supply rail. Unclipped value would have been {result['vout_ideal_unclipped']:.2f} V -- lower a fader or input level.")

    st.divider()
    st.subheader("Output waveform (combined signal)")

    t = np.linspace(0, 4 * np.pi, 400)
    combined_wave = np.zeros_like(t)
    for i in range(3):
        freq_mult = i + 1
        wave = vins[i] * faders[i] * np.sin(t * freq_mult)
        combined_wave += wave / mixer.rin_list[i]

    vout_wave = np.clip(-mixer.rf * combined_wave, vee, vcc)

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(t, vout_wave, color="#D85A30", linewidth=2)
    ax2.axhline(vcc, color="gray", linestyle="--", alpha=0.4)
    ax2.axhline(vee, color="gray", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Time (arbitrary units)")
    ax2.set_ylabel("Vout (V)")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

with right_col:
    st.subheader("Circuit schematic")
    st.caption("Live schematic reflecting current resistor values and fader positions")
    st.components.v1.html(draw_circuit_svg([rin1, rin2, rin3], rf, faders), height=340)