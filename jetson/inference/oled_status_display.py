#!/usr/bin/env python3
import errno
import fcntl
import os
from typing import Dict, List, Optional

I2C_SLAVE = 0x0703


class OledStatusDisplay:
    GLYPH_WIDTH = 5
    GLYPH_SPACING = 1
    GLYPH_HEIGHT = 7
    TEXT_WRAP_COLUMNS = 16
    FONT: Dict[str, List[int]] = {
        " ": [0x00, 0x00, 0x00, 0x00, 0x00],
        "!": [0x00, 0x00, 0x5F, 0x00, 0x00],
        "-": [0x08, 0x08, 0x08, 0x08, 0x08],
        "0": [0x3E, 0x51, 0x49, 0x45, 0x3E],
        "1": [0x00, 0x42, 0x7F, 0x40, 0x00],
        "2": [0x42, 0x61, 0x51, 0x49, 0x46],
        "3": [0x21, 0x41, 0x45, 0x4B, 0x31],
        "4": [0x18, 0x14, 0x12, 0x7F, 0x10],
        "5": [0x27, 0x45, 0x45, 0x45, 0x39],
        "6": [0x3C, 0x4A, 0x49, 0x49, 0x30],
        "7": [0x01, 0x71, 0x09, 0x05, 0x03],
        "8": [0x36, 0x49, 0x49, 0x49, 0x36],
        "9": [0x06, 0x49, 0x49, 0x29, 0x1E],
        "A": [0x7E, 0x11, 0x11, 0x11, 0x7E],
        "B": [0x7F, 0x49, 0x49, 0x49, 0x36],
        "C": [0x3E, 0x41, 0x41, 0x41, 0x22],
        "D": [0x7F, 0x41, 0x41, 0x22, 0x1C],
        "E": [0x7F, 0x49, 0x49, 0x49, 0x41],
        "F": [0x7F, 0x09, 0x09, 0x09, 0x01],
        "G": [0x3E, 0x41, 0x49, 0x49, 0x7A],
        "H": [0x7F, 0x08, 0x08, 0x08, 0x7F],
        "I": [0x00, 0x41, 0x7F, 0x41, 0x00],
        "J": [0x20, 0x40, 0x41, 0x3F, 0x01],
        "K": [0x7F, 0x08, 0x14, 0x22, 0x41],
        "L": [0x7F, 0x40, 0x40, 0x40, 0x40],
        "M": [0x7F, 0x02, 0x0C, 0x02, 0x7F],
        "N": [0x7F, 0x04, 0x08, 0x10, 0x7F],
        "O": [0x3E, 0x41, 0x41, 0x41, 0x3E],
        "P": [0x7F, 0x09, 0x09, 0x09, 0x06],
        "Q": [0x3E, 0x41, 0x51, 0x21, 0x5E],
        "R": [0x7F, 0x09, 0x19, 0x29, 0x46],
        "S": [0x46, 0x49, 0x49, 0x49, 0x31],
        "T": [0x01, 0x01, 0x7F, 0x01, 0x01],
        "U": [0x3F, 0x40, 0x40, 0x40, 0x3F],
        "V": [0x1F, 0x20, 0x40, 0x20, 0x1F],
        "W": [0x7F, 0x20, 0x18, 0x20, 0x7F],
        "X": [0x63, 0x14, 0x08, 0x14, 0x63],
        "Y": [0x07, 0x08, 0x70, 0x08, 0x07],
        "Z": [0x61, 0x51, 0x49, 0x45, 0x43],
    }

    def __init__(self, enabled: bool, i2c_device: str = "/dev/i2c-7", i2c_address: int = 0x3C,
                 width: int = 128, height: int = 64, column_offset: int = 0,
                 text_scale: int = 2) -> None:
        self.enabled = enabled
        self.i2c_device = i2c_device
        self.i2c_address = i2c_address
        self.width = width
        self.height = height
        self.column_offset = column_offset
        self.text_scale = max(int(text_scale), 1)
        self.fd: Optional[int] = None
        self.last_text: Optional[str] = None
        if not self.enabled:
            return
        self._open()
        self.initialize()
        self.clear()

    def _open(self) -> None:
        if os.name != "posix":
            raise RuntimeError("OLED display is only supported on Linux")
        try:
            self.fd = os.open(self.i2c_device, os.O_RDWR)
        except OSError as exc:
            raise RuntimeError(f"failed to open {self.i2c_device}: {exc.strerror}") from exc
        try:
            fcntl.ioctl(self.fd, I2C_SLAVE, self.i2c_address)
        except OSError as exc:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError(
                f"failed to select OLED address 0x{self.i2c_address:02x}: {exc.strerror}"
            ) from exc

    def close(self) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def page_count(self) -> int:
        return self.height // 8

    def _write_all(self, payload: bytes) -> None:
        if self.fd is None:
            raise RuntimeError("OLED display is not open")
        view = memoryview(payload)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise OSError(errno.EIO, "short OLED write")
            view = view[written:]

    def command(self, values: List[int]) -> None:
        self._write_all(bytes([0x00] + values))

    def data(self, values: List[int]) -> None:
        self._write_all(bytes([0x40] + values))

    def initialize(self) -> None:
        self.command([
            0xAE, 0xD5, 0x80, 0xA8, self.height - 1, 0xD3, 0x00, 0x40,
            0x8D, 0x14, 0x20, 0x00, 0xA1, 0xC8, 0xDA,
            0x12 if self.height == 64 else 0x02, 0x81, 0xCF, 0xD9, 0xF1,
            0xDB, 0x40, 0xA4, 0xA6, 0x2E, 0xAF,
        ])

    def clear(self) -> None:
        self.show([0x00] * (self.width * self.page_count()))
        self.last_text = None

    def show(self, framebuffer: List[int]) -> None:
        pages = self.page_count()
        if len(framebuffer) != self.width * pages:
            raise RuntimeError("invalid OLED framebuffer size")
        for page in range(pages):
            start = self.column_offset
            end = self.column_offset + self.width - 1
            self.command([
                0xB0 + page,
                start & 0x0F,
                0x10 | (start >> 4),
                0x21,
                start,
                end,
            ])
            row = framebuffer[page * self.width : (page + 1) * self.width]
            for index in range(0, self.width, 16):
                self.data(row[index : index + 16])

    def normalize_char(self, char: str) -> str:
        return char.upper()

    def wrap_text(self, text: str) -> List[str]:
        lines: List[str] = []
        current = ""
        words = text.split()
        for word in words:
            next_size = len(word) if not current else len(current) + 1 + len(word)
            if next_size > self.TEXT_WRAP_COLUMNS and current:
                lines.append(current)
                current = word
            else:
                current = word if not current else f"{current} {word}"
        if current:
            lines.append(current)
        if not lines:
            lines.append("")
        return lines

    def set_pixel(self, framebuffer: List[int], x: int, y: int) -> None:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return
        page = y // 8
        bit = y % 8
        framebuffer[page * self.width + x] |= 1 << bit

    def draw_char(self, framebuffer: List[int], x: int, y: int, char: str) -> None:
        glyph = self.FONT.get(self.normalize_char(char), self.FONT[" "])
        for col in range(self.GLYPH_WIDTH):
            for row in range(self.GLYPH_HEIGHT):
                if ((glyph[col] >> row) & 0x1) == 0:
                    continue
                for scale_x in range(self.text_scale):
                    for scale_y in range(self.text_scale):
                        self.set_pixel(
                            framebuffer,
                            x + col * self.text_scale + scale_x,
                            y + row * self.text_scale + scale_y,
                        )

    def render_text(self, text: str) -> List[int]:
        framebuffer = [0x00] * (self.width * self.page_count())
        lines = self.wrap_text(text)
        scaled_glyph_height = self.GLYPH_HEIGHT * self.text_scale
        scaled_glyph_width = self.GLYPH_WIDTH * self.text_scale
        scaled_spacing = self.GLYPH_SPACING * self.text_scale
        line_gap = max(3, self.text_scale * 2)
        total_height = len(lines) * (scaled_glyph_height + line_gap)
        y = max(0, (self.height - total_height) // 2)
        for line in lines:
            line_width = len(line) * (scaled_glyph_width + scaled_spacing) - (
                0 if not line else scaled_spacing
            )
            x = max(0, (self.width - line_width) // 2)
            for char in line:
                self.draw_char(framebuffer, x, y, char)
                x += scaled_glyph_width + scaled_spacing
            y += scaled_glyph_height + line_gap
        return framebuffer

    def write_text(self, text: str) -> None:
        if not self.enabled:
            return
        if text == self.last_text:
            return
        self.show(self.render_text(text))
        self.last_text = text
