package org.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

public class Statistics {
    private final DataClassifier classifier;

    public Statistics(DataClassifier classifier) {
        this.classifier = classifier;
    }

    public void printStatistics(StatsMode statsMode) {
        switch (statsMode) {
            case SHORT -> printShortStatistics();
            case FULL -> printFullStatistics();
        }
    }

    public void printFileStatistics(Path outputPath, String prefix, StatsMode statsMode) {
        Set<Integer> integers = readDataFromFile(outputPath, prefix + "integers.txt", Integer::parseInt);
        Set<Float> floats = readDataFromFile(outputPath, prefix + "floats.txt", Float::parseFloat);
        Set<String> strings = readDataFromFile(outputPath, prefix + "strings.txt", Function.identity());

        switch (statsMode) {
            case SHORT -> printShortStatistics(integers, floats, strings);
            case FULL -> printFullStatistics(integers, floats, strings);
        }
    }

    private void printShortStatistics() {
        printShortStatistics(classifier.getIntegers(), classifier.getFloats(), classifier.getStrings());
    }

    private void printShortStatistics(Set<Integer> integers, Set<Float> floats, Set<String> strings) {
        System.out.println("Краткая статистика:");
        System.out.printf("Целые числа: %d%nДробные числа: %d%nСтроки: %d%n", integers.size(), floats.size(), strings.size());
    }

    private void printFullStatistics() {
        printFullStatistics(classifier.getIntegers(), classifier.getFloats(), classifier.getStrings());
    }

    private void printFullStatistics(Set<Integer> integers, Set<Float> floats, Set<String> strings) {
        System.out.println("Полная статистика:");
        printNumberStatistics("Целые числа", integers);
        printNumberStatistics("Дробные числа", floats);
        printStringStatistics(strings);
    }

    private <T extends Number & Comparable<T>> void printNumberStatistics(String label, Set<T> numbers) {
        if (numbers.isEmpty()) return;

        T min = Collections.min(numbers);
        T max = Collections.max(numbers);
        double sum = numbers.stream().mapToDouble(Number::doubleValue).sum();
        double average = numbers.stream().mapToDouble(Number::doubleValue).average().orElse(0);

        System.out.printf("%s:%n  Количество: %d%n  Минимальное: %s%n  Максимальное: %s%n  Сумма: %.2f%n  Среднее: %.2f%n",
                label, numbers.size(), min, max, sum, average);
    }

    private void printStringStatistics(Set<String> strings) {
        if (strings.isEmpty()) return;

        int minLength = strings.stream().mapToInt(String::length).min().orElse(0);
        int maxLength = strings.stream().mapToInt(String::length).max().orElse(0);

        System.out.printf("Строки:%n  Количество: %d%n  Самая короткая строка: %d символов%n  Самая длинная строка: %d символов%n",
                strings.size(), minLength, maxLength);
    }

    private <T> Set<T> readDataFromFile(Path outputPath, String fileName, Function<String, T> parser) {
        Path filePath = outputPath.resolve(fileName);
        if (!Files.exists(filePath)) {
            return Collections.emptySet();
        }
        try {
            return Files.readAllLines(filePath).stream().map(parser).collect(Collectors.toCollection(LinkedHashSet::new));
        } catch (IOException | NumberFormatException e) {
            System.err.println("Ошибка при чтении файла " + fileName + ": " + e.getMessage());
            return Collections.emptySet();
        }
    }
}