#!/usr/bin/env python3
import datetime
import os
import wave
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
MPL_CONFIG_DIR = OUTPUT_DIR / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
os.environ.setdefault("MPLBACKEND", "TkAgg")

# Matplotlib will fall back to an interactive backend if one is available.
# Users can choisir un backend spécifique en définissant MPLBACKEND (ex : QtAgg, TkAgg).
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

try:
    import sounddevice as sd
except ImportError:  # sounddevice remains optional
    sd = None

try:
    import simpleaudio as sa
except ImportError:  # simpleaudio remains optional
    sa = None

DURATION_SECONDS = 1.0
SAMPLE_RATE = 44_100
time_axis = np.linspace(
    0.0, DURATION_SECONDS, int(SAMPLE_RATE * DURATION_SECONDS), endpoint=False
)

INIT_AMPLITUDE = 1.0
INIT_FREQUENCY = 440.0
INIT_PHASE = 0.0
INIT_OFFSET = 0.0
INIT_DAMPING = 0.0
INIT_NOISE = 0.0

rng = np.random.default_rng(42)
noise_profile = rng.standard_normal(time_axis.shape)

fig, ax = plt.subplots(figsize=(12, 7))
plt.subplots_adjust(left=0.07, right=0.97, bottom=0.42, top=0.92)
line_dry, = ax.plot(time_axis, np.zeros_like(time_axis), lw=2, label="Signal principal")
line_impulse, = ax.plot(
    time_axis, np.zeros_like(time_axis), lw=1.5, label="Sinusoïde réverb", color="C1"
)
line_wet, = ax.plot(
    time_axis, np.zeros_like(time_axis), lw=2, label="Signal convolué", color="C2"
)
ax.set_title("Sinusoïde réglable")
ax.set_xlabel("Temps (s)")
ax.set_ylabel("Amplitude")
ax.grid(True)

main_slider_defs = [
    ("Amplitude", 0.0, 5.0, INIT_AMPLITUDE),
    ("Fréquence (Hz)", 0.0, 10_000.0, INIT_FREQUENCY),
    ("Phase (rad)", 0.0, 2 * np.pi, INIT_PHASE),
    ("Offset", -5.0, 5.0, INIT_OFFSET),
    ("Damping", 0.0, 1.0, INIT_DAMPING),
    ("Bruit", 0.0, 1.0, INIT_NOISE),
]

reverb_slider_defs = [
    ("Réverb Amp", 0.0, 2.0, 0.5),
    ("Réverb Fréq (Hz)", 0.0, 5_000.0, 300.0),
    ("Réverb Phase", 0.0, 2 * np.pi, 0.0),
    ("Réverb Damping", 0.0, 2.0, 0.8),
    ("Mix", 0.0, 1.0, 0.3),
]

SLIDER_HEIGHT = 0.022
SLIDER_STEP = 0.042
MAIN_SLIDER_X = 0.07
REVERB_SLIDER_X = 0.45
SLIDER_TOP = 0.40
SLIDER_WIDTH = 0.2
TEXTBOX_WIDTH = 0.045
TEXTBOX_GAP = 0.018

slider_textbox_pairs = []


def format_slider_value(value: float) -> str:
    return f"{value:.6g}"


def set_textbox_value(textbox: TextBox, value: str) -> None:
    previous_state = textbox.eventson
    textbox.eventson = False
    textbox.set_val(value)
    textbox.eventson = previous_state


main_sliders = []
for idx, (label, vmin, vmax, vinit) in enumerate(main_slider_defs):
    bottom = SLIDER_TOP - idx * SLIDER_STEP
    ax_slider = plt.axes([MAIN_SLIDER_X, bottom, SLIDER_WIDTH, SLIDER_HEIGHT])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit)
    ax_slider.set_facecolor("#fbfbfb")
    main_sliders.append(slider)

    text_x = MAIN_SLIDER_X + SLIDER_WIDTH + TEXTBOX_GAP
    ax_text = plt.axes([text_x, bottom, TEXTBOX_WIDTH, SLIDER_HEIGHT])
    textbox = TextBox(ax_text, "", initial=format_slider_value(vinit))
    ax_text.set_facecolor("#f2f2f2")
    textbox.text_disp.set_ha("right")
    textbox.text_disp.set_va("center")
    slider_textbox_pairs.append((slider, textbox))

    def _make_submit(s=slider, tb=textbox):
        def _submit(text: str):
            cleaned = text.replace(",", ".").strip()
            if not cleaned:
                cleaned = "0"
            try:
                value = float(cleaned)
            except ValueError:
                set_textbox_value(tb, format_slider_value(s.val))
                return
            clipped = np.clip(value, s.valmin, s.valmax)
            s.set_val(clipped)
            set_textbox_value(tb, format_slider_value(s.val))

        return _submit

    textbox.on_submit(_make_submit())

reverb_sliders = []
for idx, (label, vmin, vmax, vinit) in enumerate(reverb_slider_defs):
    bottom = SLIDER_TOP - idx * SLIDER_STEP
    ax_slider = plt.axes([REVERB_SLIDER_X, bottom, SLIDER_WIDTH, SLIDER_HEIGHT])
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit)
    ax_slider.set_facecolor("#fbfbfb")
    reverb_sliders.append(slider)

    text_x = REVERB_SLIDER_X + SLIDER_WIDTH + TEXTBOX_GAP
    ax_text = plt.axes([text_x, bottom, TEXTBOX_WIDTH, SLIDER_HEIGHT])
    textbox = TextBox(ax_text, "", initial=format_slider_value(vinit))
    ax_text.set_facecolor("#f2f2f2")
    textbox.text_disp.set_ha("right")
    textbox.text_disp.set_va("center")
    slider_textbox_pairs.append((slider, textbox))

    def _make_submit(s=slider, tb=textbox):
        def _submit(text: str):
            cleaned = text.replace(",", ".").strip()
            if not cleaned:
                cleaned = "0"
            try:
                value = float(cleaned)
            except ValueError:
                set_textbox_value(tb, format_slider_value(s.val))
                return
            clipped = np.clip(value, s.valmin, s.valmax)
            s.set_val(clipped)
            set_textbox_value(tb, format_slider_value(s.val))

        return _submit

    textbox.on_submit(_make_submit())

waveform_labels = ["Sinus", "Carré", "Triangle"]
waveform_main_choice = waveform_labels[0]
waveform_reverb_choice = waveform_labels[0]

radio_main_ax = plt.axes([0.07, 0.12, 0.15, 0.10])
radio_reverb_ax = plt.axes([0.28, 0.12, 0.15, 0.10])
radio_main = RadioButtons(radio_main_ax, waveform_labels, active=0)
radio_reverb = RadioButtons(radio_reverb_ax, waveform_labels, active=0)

visibility_ax = plt.axes([0.52, 0.10, 0.26, 0.14])
visibility_buttons = CheckButtons(
    visibility_ax,
    ("Signal principal", "Sinusoïde réverb", "Signal convolué"),
    (True, True, True),
)
radio_main_ax.set_facecolor("#f9f9f9")
radio_reverb_ax.set_facecolor("#f9f9f9")
visibility_ax.set_facecolor("#f9f9f9")

for slider in main_sliders + reverb_sliders:
    slider.label.set_fontsize(8)
    slider.valtext.set_visible(False)

for label in radio_main.labels:
    label.set_fontsize(8)

for label in radio_reverb.labels:
    label.set_fontsize(8)

for label in visibility_buttons.labels:
    label.set_fontsize(8)
    label.set_x(0.1)

for radio in (radio_main, radio_reverb):
    circles = getattr(radio, "circles", None)
    if not circles:
        continue
    for circle in circles:
        circle.radius = 0.04

lines_by_label = {
    "Signal principal": line_dry,
    "Sinusoïde réverb": line_impulse,
    "Signal convolué": line_wet,
}
legend = ax.legend(loc="upper right")
legend_handles_seq = getattr(legend, "legend_handles", None)
if legend_handles_seq is None:
    legend_handles_seq = getattr(legend, "legendHandles", [])
legend_handles = {
    line.get_label(): handle
    for line, handle in zip(
        (line_dry, line_impulse, line_wet),
        legend_handles_seq,
    )
}


def generate_waveform(kind: str, amplitude: float, frequency: float, phase: float) -> np.ndarray:
    base_phase = 2 * np.pi * frequency * time_axis + phase
    sine_wave = np.sin(base_phase)
    if kind == "Carré":
        base_wave = np.where(sine_wave >= 0.0, 1.0, -1.0)
    elif kind == "Triangle":
        base_wave = (2.0 / np.pi) * np.arcsin(sine_wave)
    else:
        base_wave = sine_wave
    return amplitude * base_wave


def compute_signals():
    a, f, p, o, d, n = [s.val for s in main_sliders]
    rev_amp, rev_freq, rev_phase, rev_damping, mix = [s.val for s in reverb_sliders]

    envelope = np.exp(-d * (time_axis - time_axis[0]))
    dry_wave = generate_waveform(waveform_main_choice, a, f, p)
    dry = envelope * dry_wave + o
    if n > 0:
        dry = dry + n * noise_profile

    rev_envelope = np.exp(-rev_damping * (time_axis - time_axis[0]))
    impulse_wave = generate_waveform(waveform_reverb_choice, rev_amp, rev_freq, rev_phase)
    impulse = rev_envelope * impulse_wave

    wet = apply_convolution_reverb(dry, impulse, mix)
    return dry, impulse, wet


def apply_convolution_reverb(
    signal: np.ndarray, impulse: np.ndarray, mix: float
) -> np.ndarray:
    if mix <= 0 or np.allclose(impulse, 0.0):
        return signal
    convolution = np.convolve(signal, impulse, mode="full")
    trimmed = convolution[: signal.size]
    wet = normalize_waveform(trimmed)
    return (1.0 - mix) * signal + mix * wet


def update(_):
    dry, impulse, wet = compute_signals()
    line_dry.set_ydata(dry)
    line_impulse.set_ydata(impulse)
    line_wet.set_ydata(wet)

    visible_data = [
        line.get_ydata()
        for line in (line_dry, line_impulse, line_wet)
        if line.get_visible()
    ]
    if not visible_data:
        visible_data = [np.zeros_like(time_axis)]
    stacked = np.vstack(visible_data)
    y_min = stacked.min()
    y_max = stacked.max()
    padding = 0.1 * (y_max - y_min if y_max != y_min else 1.0)
    ax.set_ylim(y_min - padding, y_max + padding)

    freq = main_sliders[1].val
    if freq > 0:
        period = 1.0 / freq
        window = min(time_axis[-1], 2.0 * period)
        window = max(window, time_axis[1] - time_axis[0])
        ax.set_xlim(time_axis[0], time_axis[0] + window)
    else:
        ax.set_xlim(time_axis[0], time_axis[-1])

    fig.canvas.draw_idle()


def _make_slider_change_handler(slider, textbox):
    def _handler(val):
        set_textbox_value(textbox, format_slider_value(val))
        update(None)

    return _handler


for slider, textbox in slider_textbox_pairs:
    slider.on_changed(_make_slider_change_handler(slider, textbox))


def on_waveform_main(label: str):
    global waveform_main_choice
    waveform_main_choice = label
    update(None)


def on_waveform_reverb(label: str):
    global waveform_reverb_choice
    waveform_reverb_choice = label
    update(None)


def on_visibility_change(label: str):
    line = lines_by_label[label]
    new_state = not line.get_visible()
    line.set_visible(new_state)
    handle = legend_handles.get(label)
    if handle is not None:
        handle.set_alpha(1.0 if new_state else 0.2)
    update(None)
    fig.canvas.draw_idle()


def timestamped_path(extension: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"sinusoide_{stamp}.{extension}"


def export_csv(_):
    path = timestamped_path("csv")
    dry, impulse, wet = compute_signals()
    np.savetxt(
        path,
        np.column_stack((time_axis, dry, impulse, wet)),
        delimiter=",",
        header="t,dry,impulse,wet",
        comments="",
    )
    print(f"Courbe exportée en CSV -> {path}")


def export_wav(_):
    path = timestamped_path("wav")
    _, _, wet = compute_signals()
    pcm16 = waveform_to_pcm16(wet)
    with path.open("wb") as file_obj:
        with wave.open(file_obj, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(pcm16.tobytes())
    print(f"Signal exporté en WAV -> {path}")


def normalize_waveform(data: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(data))
    return data if peak == 0 else data / peak


def waveform_to_pcm16(data: np.ndarray) -> np.ndarray:
    normalized = normalize_waveform(data)
    pcm16 = np.clip(normalized, -1, 1)
    return (pcm16 * 32767).astype(np.int16)


current_playback = None
active_backend = None
USE_SIMPLEAUDIO = bool(int(os.environ.get("USE_SIMPLEAUDIO", "0")))


def stop_playback():
    global current_playback, active_backend
    if active_backend == "sounddevice":
        if sd is not None:
            sd.stop()
    elif active_backend == "simpleaudio":
        if current_playback is not None:
            current_playback.stop()
    current_playback = None
    active_backend = None


def play_wave(_):
    global current_playback, active_backend
    _, _, waveform = compute_signals()
    stop_playback()

    if sd is not None:
        try:
            float_wave = normalize_waveform(waveform).astype(np.float32)
            sd.play(float_wave, SAMPLE_RATE, blocking=False)
            active_backend = "sounddevice"
            return
        except Exception as err:
            print(f"Lecture avec sounddevice impossible : {err}")

    if sa is not None and USE_SIMPLEAUDIO:
        pcm16 = waveform_to_pcm16(waveform)
        try:
            current_playback = sa.play_buffer(pcm16.tobytes(), 1, 2, SAMPLE_RATE)
            active_backend = "simpleaudio"
            return
        except Exception as err:
            current_playback = None
            print(f"Lecture avec simpleaudio impossible : {err}")

    if sd is None:
        msg = (
            "Lecture impossible : installez sounddevice (`pip install sounddevice`)."
            if not USE_SIMPLEAUDIO or sa is None
            else "Lecture impossible : installez sounddevice (`pip install sounddevice`) "
            "ou activez simpleaudio via USE_SIMPLEAUDIO=1."
        )
        print(msg)


BUTTON_HEIGHT = 0.038
BUTTON_WIDTH = 0.16
BUTTON_Y = 0.06
button_play_ax = plt.axes([0.07, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])
button_csv_ax = plt.axes([0.30, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])
button_wav_ax = plt.axes([0.53, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])
button_stop_ax = plt.axes([0.76, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT])

button_play = Button(button_play_ax, "Lire")
button_csv = Button(button_csv_ax, "Exporter CSV")
button_wav = Button(button_wav_ax, "Exporter WAV")
button_stop = Button(button_stop_ax, "Stop")
for ax_btn in (button_play_ax, button_csv_ax, button_wav_ax, button_stop_ax):
    ax_btn.set_facecolor("#f0f0f0")

for button in (button_play, button_csv, button_wav, button_stop):
    button.label.set_fontsize(10)

button_play.on_clicked(play_wave)
button_csv.on_clicked(export_csv)
button_wav.on_clicked(export_wav)
button_stop.on_clicked(lambda _: stop_playback())
radio_main.on_clicked(on_waveform_main)
radio_reverb.on_clicked(on_waveform_reverb)
visibility_buttons.on_clicked(on_visibility_change)

def on_key_press(event):
    if event.key in (" ", "space"):
        play_wave(None)


fig.canvas.mpl_connect("key_press_event", on_key_press)

update(None)
plt.show()
