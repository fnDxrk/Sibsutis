package org.example;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Set;

public class FileProcessor {
    private final Arguments arguments;
    private final DataClassifier classifier;
    private final Statistics statistics;

    public FileProcessor(Arguments arguments) {
        this.arguments = arguments;
        this.classifier = new DataClassifier();
        this.statistics = new Statistics(classifier);
    }

    public void processFiles() throws IOException {
        createOutputDirectory(arguments.getOutputPath());
        for (String fileName : arguments.getInputFiles()) {
            File file = new File(fileName);
            if (!file.exists()) {
                System.err.println("Ошибка! Файл " + fileName + " не найден.");
                continue;
            }
            processFile(file);
            writeResults();
        }
        statistics.printStatistics(arguments.getStatsMode());
    }

    private void processFile(File file) {
        try {
            List<String> lines = Files.readAllLines(file.toPath());
            for (String line : lines) {
                classifier.classify(line);
            }
        } catch (IOException e) {
            System.err.println("Ошибка при чтении файла " + file.getName() + ": " + e.getMessage());
        }
    }

    private void createOutputDirectory(String outputPath) throws IOException {
        File outputDir = new File(outputPath);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("Не удалось создать директорию: " + outputPath);
        }
    }

    private void writeResults() throws IOException {
        writeDataToFile("integers.txt", classifier.getIntegers());
        writeDataToFile("floats.txt", classifier.getFloats());
        writeDataToFile("strings.txt", classifier.getStrings());
    }

    private <T> void writeDataToFile(String fileName, Set<T> data) throws IOException {
        if (data.isEmpty()) return;

        Path filePath = Path.of(arguments.getOutputPath(), arguments.getPrefix() + fileName);
        StandardOpenOption[] options = arguments.isAppendEnabled()
                ? new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.APPEND}
                : new StandardOpenOption[]{StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING};

        List<String> lines = data.stream().map(Object::toString).toList();
        try {
            Files.write(filePath, lines, options);
        } catch (IOException e) {
            System.err.println("Ошибка при записи в файл " + filePath + ": " + e.getMessage());
        }
    }
}
