"""The raster forge — same footprint, same promises, more pixels.

``PixelForge`` sends a real PNG through the kitty or iTerm2 inline-image
protocol instead of drawing the hammer with text cells. That is a larger change
than it looks: it writes bytes no terminal is obliged to understand, into the
same stream the text display uses. So the tests that matter are not about how
the anvil looks — they are about whether the existing contracts still hold once
a picture is involved.

The four that would each be a real defect:

* a terminal without an inline-graphics protocol must see exactly what it saw
  before, so the raster backend is safe to construct unconditionally;
* nothing may reach stdout, ever, because a ``--format json`` consumer is still
  on the other end of it;
* the scroll region must still be released — a graphics payload gives the exit
  path more chances to fail before it gets there;
* the placed image must be deleted, or it outlives the run on the terminal.
"""
import io
import struct
import subprocess
import sys

import pytest

from tarhan import cliout
from tarhan.forge_pixels import (GRAPHICS_MODES, PIXEL_HEIGHT, PIXEL_WIDTH,
                                 PixelForge, detect_graphics, render_sprite)


class FakeTTY(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


def _drive(forge):
    with forge:
        forge.begin("SOLVE", "newton")
        forge.tick()
        forge.tick()
        forge.converged("done")


# --- the sprite is a real PNG --------------------------------------------

@pytest.mark.parametrize("frame", range(4))
def test_each_frame_is_a_valid_png_of_the_declared_size(frame):
    """The encoder is hand-rolled from zlib and struct, so its output is
    checked as bytes rather than trusted."""
    png = render_sprite(frame)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert png[12:16] == b"IHDR"
    width, height = struct.unpack(">II", png[16:24])
    assert (width, height) == (PIXEL_WIDTH, PIXEL_HEIGHT)
    # An IEND chunk is length(4) + type(4) + no data + crc(4). The type sits at
    # [-8:-4], not [-12:-8] — that is the zero length field, and reading it as
    # the type is how this assertion was first written and first failed.
    assert png[-8:-4] == b"IEND"
    assert png[-12:-8] == b"\x00\x00\x00\x00"

    # Walk the chunks so a malformed length cannot hide behind the endpoints.
    offset, kinds = 8, []
    while offset < len(png):
        size = struct.unpack(">I", png[offset:offset + 4])[0]
        kinds.append(png[offset + 4:offset + 8])
        offset += 12 + size
    assert offset == len(png), "a chunk length does not agree with the file size"
    assert kinds == [b"IHDR", b"IDAT", b"IEND"]


def test_the_strike_frame_differs_from_the_others():
    """A four-frame cycle whose frames are identical is not an animation."""
    frames = [render_sprite(i) for i in range(4)]
    assert frames[2] != frames[0]
    assert frames[2] != frames[1]


def test_the_frame_index_wraps():
    assert render_sprite(6) == render_sprite(2)


# --- protocol detection ---------------------------------------------------

@pytest.mark.parametrize("env,expected", [
    ({"KITTY_WINDOW_ID": "3"}, "kitty"),
    ({"TERM": "xterm-kitty"}, "kitty"),
    ({"TERM_PROGRAM": "iTerm.app"}, "iterm"),
    ({"LC_TERMINAL": "iTerm2"}, "iterm"),
    ({"TERM": "xterm-256color"}, None),
    ({}, None),
])
def test_detection_only_claims_a_protocol_it_can_identify(env, expected):
    """Guessing wrong means writing image bytes at a terminal that will print
    them as garbage, so absence of evidence has to mean None."""
    assert detect_graphics(env) is expected


def test_an_unknown_graphics_mode_is_refused():
    out = cliout.Output(stderr=FakeTTY())
    with pytest.raises(ValueError, match="graphics must be one of"):
        PixelForge(["A"], out, graphics="sixel")
    assert set(GRAPHICS_MODES) == {"auto", "kitty", "iterm", "text"}


# --- the contracts that already existed -----------------------------------

def test_a_pipe_sees_no_image_bytes_at_all():
    """The invariant defended all session, re-checked with a raster backend.

    A StringIO is not a terminal, so the display must stay silent — and in
    particular must not emit a protocol payload that a log would carry forever.
    """
    out = cliout.Output(stdout=io.StringIO(), stderr=io.StringIO())
    forge = PixelForge(["SOLVE"], out, graphics="kitty")
    assert forge.animate is False
    _drive(forge)
    written = out.stderr.getvalue()
    assert out.stdout.getvalue() == ""
    assert "_Ga=" not in written
    assert "]1337;" not in written


def test_nothing_reaches_stdout_even_when_drawing_images():
    out = cliout.Output(color="always", stdout=io.StringIO(), stderr=FakeTTY())
    _drive(PixelForge(["SOLVE"], out, graphics="kitty", pin=True))
    assert out.stdout.getvalue() == ""


def test_the_kitty_image_is_placed_and_then_deleted():
    """An image left placed outlives the run and sits on the user's terminal."""
    out = cliout.Output(color="always", stdout=io.StringIO(), stderr=FakeTTY())
    _drive(PixelForge(["SOLVE"], out, graphics="kitty", pin=True))
    written = out.stderr.getvalue()
    assert written.count("_Ga=T") >= 1, "no image was ever placed"
    assert written.count("_Ga=d") >= written.count("_Ga=T"), \
        "every placement must be matched by a delete"


def test_the_scroll_region_is_still_released_with_graphics_in_play():
    out = cliout.Output(color="always", stdout=io.StringIO(), stderr=FakeTTY())
    _drive(PixelForge(["SOLVE"], out, graphics="kitty", pin=True))
    written = out.stderr.getvalue()
    assert "\x1b[2J" not in written
    assert "\x1b[1;" in written and "\x1b[r" in written
    assert written.count("\x1b[?25l") == written.count("\x1b[?25h")


def test_text_mode_behaves_exactly_like_the_plain_forge():
    """`--graphics text` must be a real off switch, not a hint."""
    out = cliout.Output(color="always", stdout=io.StringIO(), stderr=FakeTTY())
    forge = PixelForge(["SOLVE"], out, graphics="text", pin=True)
    assert forge.graphics_available is False
    _drive(forge)
    written = out.stderr.getvalue()
    assert "_Ga=" not in written and "]1337;" not in written
    assert "▄▆▄" in written, "the text indicator should have been used instead"


# --- the CLI accepts and honours the flag --------------------------------

def test_the_cli_accepts_the_graphics_flag_and_keeps_json_clean(tmp_path):
    import json

    proc = subprocess.run(
        [sys.executable, "-m", "tarhan.cli", "--graphics", "text",
         "--format", "json", "run", "solve",
         "semiconductor.pn.drift-diffusion.1d.steady",
         "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == cliout.EXIT_OK, proc.stderr
    json.loads(proc.stdout)
    assert "_Ga=" not in proc.stdout and "_Ga=" not in proc.stderr
