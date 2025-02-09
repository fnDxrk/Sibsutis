package org.example;

import java.util.Set;

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

    private void printShortStatistics() {
        System.out.println("Краткая статистика:");
        System.out.println("Целые числа: " + classifier.getIntegers().size());
        System.out.println("Дробные числа: " + classifier.getFloats().size());
        System.out.println("Строки: " + classifier.getStrings().size());
    }

    private void printFullStatistics() {
        System.out.println("Полная статистика:");
        printIntegerStatistics();
        printFloatStatistics();
        printStringStatistics();
    }

    private void printIntegerStatistics() {
        Set<Integer> integers = classifier.getIntegers();
        if (integers.isEmpty()) return;

        int min = integers.stream().min(Integer::compare).orElse(0);
        int max = integers.stream().max(Integer::compare).orElse(0);
        int sum = integers.stream().mapToInt(Integer::intValue).sum();
        double average = integers.stream().mapToInt(Integer::intValue).average().orElse(0);

        System.out.println("Целые числа:");
        System.out.println("  Количество: " + integers.size());
        System.out.println("  Минимальное: " + min);
        System.out.println("  Максимальное: " + max);
        System.out.println("  Сумма: " + sum);
        System.out.println("  Среднее: " + average);
    }

    private void printFloatStatistics() {
        Set<Float> floats = classifier.getFloats();
        if (floats.isEmpty()) return;

        float min = floats.stream().min(Float::compare).orElse(0f);
        float max = floats.stream().max(Float::compare).orElse(0f);
        float sum = (float) floats.stream().mapToDouble(Float::doubleValue).sum();
        double average = floats.stream().mapToDouble(Float::doubleValue).average().orElse(0);

        System.out.println("Дробные числа:");
        System.out.println("  Количество: " + floats.size());
        System.out.println("  Минимальное: " + min);
        System.out.println("  Максимальное: " + max);
        System.out.println("  Сумма: " + sum);
        System.out.println("  Среднее: " + average);
    }

    private void printStringStatistics() {
        Set<String> strings = classifier.getStrings();
        if (strings.isEmpty()) return;

        int minLength = strings.stream().mapToInt(String::length).min().orElse(0);
        int maxLength = strings.stream().mapToInt(String::length).max().orElse(0);

        System.out.println("Строки:");
        System.out.println("  Количество: " + strings.size());
        System.out.println("  Самая короткая строка: " + minLength + " символов");
        System.out.println("  Самая длинная строка: " + maxLength + " символов");
    }
}
