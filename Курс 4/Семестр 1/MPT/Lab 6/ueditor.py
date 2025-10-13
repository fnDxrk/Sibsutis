class TEditor:
    # Constants
    DECIMAL_SEPARATOR = ","
    IMAGINARY_SEPARATOR = "i*"
    ZERO_REPRESENTATION = "0, i* 0,"

    def __init__(self):
        """Constructor - initializes with zero representation"""
        self._string = self.ZERO_REPRESENTATION
        self._has_decimal_real = True  # Zero representation already has comma
        self._has_decimal_imaginary = True  # Zero representation already has comma

    @property
    def string(self) -> str:
        """Read string in string format (property method)"""
        return self._string

    @string.setter
    def string(self, value: str) -> None:
        """Write string in string format (property method)"""
        self._string = value
        # Reset decimal flags when setting string
        parts = value.split(self.IMAGINARY_SEPARATOR)
        if len(parts) > 0:
            self._has_decimal_real = self.DECIMAL_SEPARATOR in parts[0]
        if len(parts) > 1:
            self._has_decimal_imaginary = self.DECIMAL_SEPARATOR in parts[1]
        else:
            self._has_decimal_imaginary = False

    def is_zero(self) -> bool:
        """Check if the number is complex zero"""
        return self._string == self.ZERO_REPRESENTATION

    def add_sign(self) -> str:
        """Add or remove minus sign from the string"""
        if self.is_zero():
            # For zero, just add sign to real part
            self._string = "-" + self._string
            return self._string

        parts = self._string.split(self.IMAGINARY_SEPARATOR)
        if len(parts) != 2:
            return self._string

        real_part, imaginary_part = parts

        # Toggle sign for real part
        real_part = real_part.strip()
        if real_part.startswith("-"):
            real_part = real_part[1:].strip()
        else:
            real_part = "-" + real_part

        # Reconstruct the string
        self._string = (
            f"{real_part} {self.IMAGINARY_SEPARATOR} {imaginary_part.strip()}"
        )
        return self._string

    def add_digit(self, digit: int) -> str:
        """Add a digit to the string if format allows"""
        if not 0 <= digit <= 9:
            raise ValueError("Digit must be between 0 and 9")

        if self.is_zero():
            # Replace zero with digit
            self._string = f"{digit}, i* 0,"
            self._has_decimal_real = True
            return self._string

        # Parse current complex number
        parts = self._string.split(self.IMAGINARY_SEPARATOR)
        if len(parts) != 2:
            return self._string

        real_part, imaginary_part = parts
        real_part = real_part.strip()
        imaginary_part = imaginary_part.strip()

        # Remove comma from real part to build the number
        if "," in real_part:
            real_part = real_part.replace(",", "")

        # Add digit to real part
        real_part += str(digit)

        # Add comma back to the end of real part
        real_part += ","
        self._has_decimal_real = True

        # Ensure imaginary part has comma
        if not self._has_decimal_imaginary:
            imaginary_part += ","
            self._has_decimal_imaginary = True

        self._string = f"{real_part} {self.IMAGINARY_SEPARATOR} {imaginary_part}"
        return self._string

    def add_zero(self) -> str:
        """Add zero to the string if format allows"""
        return self.add_digit(0)

    def backspace(self) -> str:
        """Remove the rightmost character"""
        if len(self._string) > 0 and not self.is_zero():
            self._string = self._string[:-1]
            # If string becomes invalid, reset to zero
            if (
                len(self._string) < len(self.ZERO_REPRESENTATION)
                or "i*" not in self._string
            ):
                self._string = self.ZERO_REPRESENTATION
                self._has_decimal_real = True
                self._has_decimal_imaginary = True
            else:
                # Update decimal flags
                parts = self._string.split(self.IMAGINARY_SEPARATOR)
                if len(parts) >= 1:
                    self._has_decimal_real = self.DECIMAL_SEPARATOR in parts[0]
                if len(parts) >= 2:
                    self._has_decimal_imaginary = self.DECIMAL_SEPARATOR in parts[1]
        return self._string

    def clear(self) -> str:
        """Set to zero complex number representation"""
        self._string = self.ZERO_REPRESENTATION
        self._has_decimal_real = True
        self._has_decimal_imaginary = True
        return self._string

    def edit(self, command: int) -> str:
        """Execute editing command based on command number"""
        commands = {
            0: self.clear,
            1: self.backspace,
            2: self.add_sign,
            3: self.add_zero,
        }

        if command in commands:
            return commands[command]()
        elif 4 <= command <= 13:  # Commands for digits 0-9
            return self.add_digit(command - 4)
        else:
            raise ValueError(f"Unknown command: {command}")

    def add_decimal_separator(self) -> str:
        """Add decimal separator to real or imaginary part"""
        if self.is_zero():
            self._string = "0, i* 0,"
            self._has_decimal_real = True
            self._has_decimal_imaginary = True
            return self._string

        parts = self._string.split(self.IMAGINARY_SEPARATOR)
        if len(parts) != 2:
            return self._string

        real_part, imaginary_part = parts
        real_part = real_part.strip()
        imaginary_part = imaginary_part.strip()

        # Add to real part if it doesn't have decimal
        if not self._has_decimal_real:
            real_part += self.DECIMAL_SEPARATOR
            self._has_decimal_real = True
        # Add to imaginary part if real part already has decimal
        elif not self._has_decimal_imaginary:
            imaginary_part += self.DECIMAL_SEPARATOR
            self._has_decimal_imaginary = True

        self._string = f"{real_part} {self.IMAGINARY_SEPARATOR} {imaginary_part}"
        return self._string

    def add_imaginary_separator(self) -> str:
        """Add imaginary separator"""
        return self._string
