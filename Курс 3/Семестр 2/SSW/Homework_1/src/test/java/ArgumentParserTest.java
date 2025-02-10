import org.example.ArgumentParser;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ArgumentParserTest {
    @Test
    void parse_ThrowsException_WhenNoArguments() {
        String[] args = {};
        Exception exception = assertThrows(IllegalArgumentException.class, () -> ArgumentParser.parse(args));
        assertEquals("Ошибка! Параметры отсутствуют.", exception.getMessage());
    }
}