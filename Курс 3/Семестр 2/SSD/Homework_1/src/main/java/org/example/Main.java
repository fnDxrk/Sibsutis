package org.example;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        try {
            Arguments arguments = parseArguments(args);

            System.out.println("Выходной путь: " + arguments.outputPath);
            System.out.println("Префикс: " + arguments.prefix);
            System.out.println("Режим добавления: " + arguments.appendEnabled);
            System.out.println("Режим статистики: " + arguments.statsMode);
            System.out.println("Файлы для обработки: " + arguments.inputFiles);
        } catch (IllegalArgumentException | IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }
    }

    private static Arguments parseArguments(String[] args) throws IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Ошибка! Параметры отсутствуют.");
        }

        List<String> inputFiles = new ArrayList<>();
        String outputPath = ".";
        String prefix = "";
        boolean appendEnabled = false;
        StatsMode statsMode = StatsMode.NONE;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-o" -> {
                    if (i + 1 < args.length) {
                        outputPath = args[++i];
                        createOutputDirectory(outputPath);
                    } else {
                        throw new IllegalArgumentException("Ошибка! Отсутствует путь после параметра -o");
                    }
                }
                case "-p" -> {
                    if (i + 1 < args.length) {
                        prefix = args[++i];
                    } else {
                        throw new IllegalArgumentException("Ошибка! Отсутствует префикс после параметра -p");
                    }
                }
                case "-a" -> appendEnabled = true;
                case "-s" -> statsMode = StatsMode.SHORT;
                case "-f" -> statsMode = StatsMode.FULL;
                default -> {
                    if (args[i].startsWith("-")) {
                        throw new IllegalArgumentException("Ошибка! Указан неверный параметр: " + args[i]);
                    }
                    inputFiles.add(args[i]);
                }
            }
        }

        if (inputFiles.isEmpty()) {
            throw new IllegalArgumentException("Ошибка! Не указаны входные файлы.");
        }

        return new Arguments(inputFiles, outputPath, prefix, appendEnabled, statsMode);
    }

    private static void createOutputDirectory(String outputPath) throws IOException {
        File outputDir = new File(outputPath);
        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("Не удалось создать директорию: " + outputPath);
        }
    }
}

enum StatsMode {
    NONE, SHORT, FULL
}

class Arguments {
    List<String> inputFiles;
    String outputPath;
    String prefix;
    boolean appendEnabled;
    StatsMode statsMode;

    public Arguments(List<String> inputFiles, String outputPath, String prefix, boolean appendEnabled, StatsMode statsMode) {
        this.inputFiles = inputFiles;
        this.outputPath = outputPath;
        this.prefix = prefix;
        this.appendEnabled = appendEnabled;
        this.statsMode = statsMode;
    }
}