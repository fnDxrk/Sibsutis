import org.example.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class FileProcessorTest {
    private Arguments arguments;
    private FileProcessor fileProcessor;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        arguments = new Arguments(List.of(), tempDir.toString(), "", false, StatsMode.NONE);
        fileProcessor = new FileProcessor(arguments);
    }

    @Test
    void processFiles_CreatesOutputDirectory() throws IOException {
        fileProcessor.processFiles();
        assertTrue(Files.exists(tempDir));
    }
}