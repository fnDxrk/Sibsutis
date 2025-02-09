package org.example;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ArgumentParser {
    public static Arguments parse(String[] args) throws IOException {
        if (args.length == 0) {
            throw new IllegalArgumentException("Ошибка! Параметры отсутствуют.");
        }

        List<String> inputFiles = new ArrayList<>();
        String outputPath = "./";
        String prefix = "";
        boolean appendEnabled = false;
        StatsMode statsMode = StatsMode.NONE;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-o" -> outputPath = getNextArg(args, i++, "Ошибка! Отсутствует путь после -o");
                case "-p" -> prefix = getNextArg(args, i++, "Ошибка! Отсутствует префикс после -p");
                case "-a" -> appendEnabled = true;
                case "-s" -> statsMode = checkStatsMode(statsMode, StatsMode.SHORT);
                case "-f" -> statsMode = checkStatsMode(statsMode, StatsMode.FULL);
                default -> handleFileArgument(args, i, inputFiles);
            }
        }

        if (inputFiles.isEmpty()) {
            throw new IllegalArgumentException("Ошибка! Не указаны входные файлы.");
        }

        return new Arguments(inputFiles, outputPath, prefix, appendEnabled, statsMode);
    }

    private static String getNextArg(String[] args, int i, String errorMessage) {
        if (i + 1 >= args.length) {
            throw new IllegalArgumentException(errorMessage);
        }
        return args[i + 1];
    }

    private static StatsMode checkStatsMode(StatsMode currentMode, StatsMode newMode) {
        if (currentMode != StatsMode.NONE) {
            throw new IllegalArgumentException("Ошибка! Нельзя указывать одновременно -s и -f.");
        }
        return newMode;
    }

    private static void handleFileArgument(String[] args, int i, List<String> inputFiles) throws IOException {
        if (args[i].startsWith("-")) {
            throw new IllegalArgumentException("Ошибка! Неизвестный параметр: " + args[i]);
        }
        if (!Files.exists(Paths.get(args[i]))) {
            throw new IllegalArgumentException("Ошибка! Файл " + args[i] + " не существует.");
        }
        inputFiles.add(args[i]);
    }
}
