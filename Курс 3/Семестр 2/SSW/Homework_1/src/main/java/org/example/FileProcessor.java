package org.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Set;

public class FileProcessor {
    private final Arguments arguments;
    private final DataClassifier classifier = new DataClassifier();
    private final Statistics statistics = new Statistics(classifier);

    public FileProcessor(Arguments arguments) {
        this.arguments = arguments;
    }

    public void processFiles() throws IOException {
        Path outputPath = Path.of(arguments.getOutputPath());
        Files.createDirectories(outputPath);

        for (String fileName : arguments.getInputFiles()) {
            Path filePath = Path.of(fileName);
            if (!Files.exists(filePath)) {
                System.err.println("Ошибка! Файл " + fileName + " не найден.");
                continue;
            }
            processFile(filePath);
        }

        clearOutputFiles(outputPath);
        writeResults(outputPath);
        statistics.printFileStatistics(outputPath, arguments.getPrefix(), arguments.getStatsMode());
    }

    private void processFile(Path filePath) {
        try {
            Files.readAllLines(filePath).forEach(classifier::classify);
        } catch (IOException e) {
            System.err.println("Ошибка при чтении файла " + filePath.getFileName() + ": " + e.getMessage());
        }
    }

    private void clearOutputFiles(Path outputPath) throws IOException {
        if (!arguments.isAppendEnabled()) {
            deleteIfExists(outputPath.resolve(arguments.getPrefix() + "integers.txt"));
            deleteIfExists(outputPath.resolve(arguments.getPrefix() + "floats.txt"));
            deleteIfExists(outputPath.resolve(arguments.getPrefix() + "strings.txt"));
        }
    }

    private void deleteIfExists(Path filePath) throws IOException {
        if (Files.exists(filePath)) {
            Files.delete(filePath);
        }
    }

    private void writeResults(Path outputPath) throws IOException {
        writeDataToFile(outputPath, "integers.txt", classifier.getIntegers());
        writeDataToFile(outputPath, "floats.txt", classifier.getFloats());
        writeDataToFile(outputPath, "strings.txt", classifier.getStrings());
    }

    private <T> void writeDataToFile(Path outputPath, String fileName, Set<T> data) throws IOException {
        if (data.isEmpty()) return;

        Path filePath = outputPath.resolve(arguments.getPrefix() + fileName);
        StandardOpenOption[] options = arguments.isAppendEnabled()
                ? new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.APPEND}
                : new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.WRITE};

        Files.write(filePath, data.stream().map(Object::toString).toList(), options);
    }
}
