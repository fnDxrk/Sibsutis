package org.example;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;

public class FileProcessor {
    private final Arguments arguments;
    private final DataClassifier classifier;

    public FileProcessor(Arguments arguments) {
        this.arguments = arguments;
        this.classifier = new DataClassifier();
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
        }
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
}
