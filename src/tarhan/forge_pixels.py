"""True-pixel, one-row forge for terminals with inline graphics support.

Seven ordinary text cells cannot carry enough solid geometry for both a hammer
and a recognisable anvil.  This experimental backend keeps the exact same
``7 columns x 1 row`` footprint but sends a 56x16 transparent PNG through the
Kitty graphics protocol or iTerm2 inline-images protocol.  Unsupported
terminals retain :class:`tarhan.forge.Forge`'s text indicator.

The anvil proportions are adapted to a tiny pixel grid from Lorc's "Anvil"
icon at game-icons.net (CC BY 3.0).  The sprite is redrawn here, not embedded.
The protocol and PNG encoder use only the Python standard library.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import os
import shutil
import struct
import time
import zlib
from typing import Mapping, Optional, Sequence, Tuple

from tarhan import cliout
from tarhan.forge import (ERASE_LINE, ESC, RESTORE_CURSOR, SAVE_CURSOR, Forge)


PIXEL_WIDTH = 56
PIXEL_HEIGHT = 16
CELL_WIDTH = 7
FRAME_COUNT = 4

RGBA = Tuple[int, int, int, int]
INK: RGBA = (218, 216, 207, 255)
HIGHLIGHT: RGBA = (245, 242, 229, 255)
SHADOW: RGBA = (167, 168, 163, 255)
HAMMER: RGBA = (211, 165, 79, 255)
HOT: RGBA = (247, 190, 70, 255)

GRAPHICS_MODES = ("auto", "kitty", "iterm", "text")
RAW_ESC = "\x1b"


def detect_graphics(environment: Optional[Mapping[str, str]] = None) \
        -> Optional[str]:
    """Return a safely identifiable inline-graphics protocol, if any."""
    env = os.environ if environment is None else environment
    term = env.get("TERM", "").lower()
    program = env.get("TERM_PROGRAM", "").lower()
    lc_terminal = env.get("LC_TERMINAL", "").lower()

    if env.get("KITTY_WINDOW_ID") or "kitty" in term:
        return "kitty"
    if program == "iterm.app" or lc_terminal == "iterm2":
        return "iterm"
    return None


class _Canvas:
    """The few raster operations this sixteen-pixel-high mark needs."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.data = bytearray(width * height * 4)

    def point(self, x: int, y: int, color: RGBA) -> None:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        offset = (y * self.width + x) * 4
        self.data[offset:offset + 4] = bytes(color)

    def span(self, y: int, x0: int, x1: int, color: RGBA) -> None:
        for x in range(x0, x1 + 1):
            self.point(x, y, color)

    def rectangle(self, x0: int, y0: int, x1: int, y1: int,
                  color: RGBA) -> None:
        for y in range(y0, y1 + 1):
            self.span(y, x0, x1, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: RGBA,
             width: int = 1) -> None:
        """Integer Bresenham line with a small square brush."""
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        radius = max(0, width // 2)
        while True:
            for py in range(y0 - radius, y0 + radius + 1):
                for px in range(x0 - radius, x0 + radius + 1):
                    self.point(px, py, color)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy


def _draw_anvil(canvas: _Canvas) -> None:
    """Face, horn, heel, shoulder and foot as separately readable masses."""
    canvas.rectangle(14, 6, 45, 9, INK)       # flat face

    # Left horn: horizontal top, then a three-step taper underneath.
    canvas.span(6, 1, 13, INK)
    canvas.span(7, 4, 13, INK)
    canvas.span(8, 7, 13, INK)
    canvas.span(9, 10, 13, INK)

    # Square heel separated from the face by a one-pixel hardy hole.
    canvas.span(6, 47, 47, INK)
    canvas.span(7, 47, 51, INK)
    canvas.span(8, 47, 55, INK)
    canvas.span(9, 47, 52, INK)
    canvas.span(10, 47, 47, INK)

    # Shoulders widen downwards, then give way to a broad rectangular foot.
    canvas.span(10, 22, 41, INK)
    canvas.span(11, 21, 42, INK)
    canvas.span(12, 20, 43, INK)
    canvas.span(13, 19, 44, INK)
    canvas.rectangle(14, 14, 48, 15, INK)

    canvas.span(6, 15, 44, HIGHLIGHT)
    canvas.span(14, 15, 47, SHADOW)


def _draw_head(canvas: _Canvas, box: Tuple[int, int, int, int],
               color: RGBA) -> None:
    """A solid hammer head with one-pixel chamfered corners."""
    x0, y0, x1, y1 = box
    canvas.span(y0, x0 + 1, x1 - 1, color)
    for y in range(y0 + 1, y1):
        canvas.span(y, x0, x1, color)
    canvas.span(y1, x0 + 1, x1 - 1, color)


def render_sprite(frame: int) -> bytes:
    """Return one 56x16 RGBA PNG frame, dependency-free."""
    frame %= FRAME_COUNT
    canvas = _Canvas(PIXEL_WIDTH, PIXEL_HEIGHT)
    _draw_anvil(canvas)

    if frame == 0:
        canvas.line(12, 3, 25, 5, HAMMER, width=2)
        _draw_head(canvas, (1, 0, 14, 4), HAMMER)
    elif frame in (1, 3):
        canvas.line(16, 4, 26, 6, HAMMER, width=2)
        _draw_head(canvas, (5, 1, 18, 5), HAMMER)
    else:
        canvas.line(19, 4, 8, 0, HAMMER, width=3)
        _draw_head(canvas, (17, 3, 30, 8), HOT)
        for x, y in ((14, 3), (12, 5), (32, 2), (34, 5), (31, 0)):
            canvas.point(x, y, HOT)
        canvas.line(13, 1, 13, 3, HOT)
        canvas.line(32, 3, 34, 3, HOT)

    return _png_rgba(PIXEL_WIDTH, PIXEL_HEIGHT, bytes(canvas.data))


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind)
    checksum = binascii.crc32(payload, checksum) & 0xffffffff
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", checksum))


def _png_rgba(width: int, height: int, pixels: bytes) -> bytes:
    stride = width * 4
    scanlines = b"".join(
        b"\x00" + pixels[y * stride:(y + 1) * stride]
        for y in range(height)
    )
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", header)
            + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
            + _png_chunk(b"IEND", b""))


def kitty_image(png: bytes, image_id: int) -> str:
    payload = base64.b64encode(png).decode("ascii")
    return (f"{RAW_ESC}_Ga=T,f=100,t=d,i={image_id},c={CELL_WIDTH},r=1,"
            f"C=1,q=2,N=1;{payload}{RAW_ESC}\\")


def kitty_delete(image_id: int) -> str:
    return f"{RAW_ESC}_Ga=d,d=I,i={image_id},q=2{RAW_ESC}\\"


def iterm_image(png: bytes) -> str:
    payload = base64.b64encode(png).decode("ascii")
    return (f"{RAW_ESC}]1337;File=inline=1;width={CELL_WIDTH};height=1;"
            f"preserveAspectRatio=0:{payload}\x07")


class PixelForge(Forge):
    """A one-line ``Forge`` using a true raster sprite when supported."""

    def __init__(self, *args, graphics: str = "auto", **kwargs) -> None:
        if graphics not in GRAPHICS_MODES:
            raise ValueError(f"graphics must be one of {GRAPHICS_MODES}")
        super().__init__(*args, **kwargs)
        detected = detect_graphics()
        self.graphics_protocol = detected if graphics == "auto" else graphics
        if self.graphics_protocol == "text":
            self.graphics_protocol = None
        self._image_id = 0x54400000 | (id(self) & 0xffff)
        self._sprite_frames: Sequence[bytes] = tuple(
            render_sprite(index) for index in range(FRAME_COUNT)
        )

    @property
    def graphics_available(self) -> bool:
        return self.graphics_protocol is not None

    def _pixel_status(self, width: int) -> str:
        stage = self._current
        name = stage.name if stage else "READY"
        elapsed = time.monotonic() - self._started
        facts = [f"{elapsed:.1f}s"]
        if self._within is not None:
            facts.append(f"{round(self._within * 100)}% of {name.lower()}")
        shown = self._index + 1 if self._index >= 0 else 1
        facts.append(f"stage {shown}/{len(self.stages)}")
        joined = self._sep.join(facts)
        tail = self._paint("(" + joined + ")", "dim")
        room = max(0, width - len(name) - len(joined) - 5)
        detail = (stage.detail if stage else "")[:room]
        return f"{self._paint(name, 'bold')}  {detail}  {tail}"

    def _image_sequence(self) -> str:
        png = self._sprite_frames[self._frame % FRAME_COUNT]
        if self.graphics_protocol == "kitty":
            return kitty_delete(self._image_id) + kitty_image(
                png, self._image_id
            )
        if self.graphics_protocol == "iterm":
            return iterm_image(png)
        return ""

    def _draw_pinned(self, text: str) -> None:
        if not self.graphics_available:
            super()._draw_pinned(text)
            return
        width = shutil.get_terminal_size((80, 24)).columns
        status = self._pixel_status(max(20, width - CELL_WIDTH - 1))
        status_column = CELL_WIDTH + 2
        self.stream.write(
            f"{SAVE_CURSOR}{ESC}{self._pin_rows};1H{ERASE_LINE}"
            f"{self._image_sequence()}"
            f"{ESC}{self._pin_rows};{status_column}H{status}"
            f"{RESTORE_CURSOR}"
        )
        self.stream.flush()

    def _release_pin(self) -> None:
        if self._pinned and self.graphics_protocol == "kitty":
            try:
                self.stream.write(kitty_delete(self._image_id))
                self.stream.flush()
            except (ValueError, OSError):
                pass
        super()._release_pin()


def demo(graphics: str = "auto") -> None:
    out = cliout.Output()
    forge = PixelForge(
        ["MESH", "ASSEMBLY", "SOLVE", "VERIFY"],
        out,
        pin=True,
        graphics=graphics,
    )
    if not forge.graphics_available:
        out.note(
            "[tarhan] inline graphics unavailable; using the text indicator"
        )
    with forge:
        forge.intro(strikes=2, pace=0.09)
        work = (
            ("MESH", "2,847 nodes / Delaunay guard", 5),
            ("ASSEMBLY", "Scharfetter-Gummel fluxes", 6),
            ("SOLVE", "damped Newton", 10),
            ("VERIFY", "charge conservation", 4),
        )
        for name, detail, count in work:
            forge.begin(name, detail)
            for step in range(count):
                time.sleep(0.2)
                forge.tick(within=(step + 1) / count)
                if name == "SOLVE" and step == 3:
                    forge.log("  note: damping engaged")
            forge.finish(detail + " ok")
        forge.converged("12 Newton iterations, residual 8.3e-11")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graphics",
        choices=GRAPHICS_MODES,
        default="auto",
        help="force a terminal graphics protocol or use text fallback",
    )
    args = parser.parse_args()
    demo(args.graphics)


if __name__ == "__main__":
    main()
