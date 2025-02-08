package org.example;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Main {
    public static void main(String[] args) {
        try {
            Arguments arguments = parseArguments(args);

//            System.out.println("Выходной путь: " + arguments.outputPath);
//            System.out.println("Префикс: " + arguments.prefix);
//            System.out.println("Режим добавления: " + arguments.appendEnabled);
//            System.out.println("Режим статистики: " + arguments.statsMode);
//            System.out.println("Файлы для обработки: " + arguments.inputFiles);

            Set<Integer> integerArrayList = new HashSet<>();
            Set<Float> floatArrayList = new HashSet<>();
            Set<String> stringArrayList = new HashSet<>();

            FileProcessor fileDataParser = new FileProcessor();

            for (String fileName : arguments.inputFiles) {
                File file = new File(fileName);
                if (!file.exists()) {
                    throw new IOException("Ошибка! Файл "  + fileName + "  не найден.");
                }
                fileDataParser.parseFile(file, integerArrayList, floatArrayList, stringArrayList);
            }

            integerArrayList.forEach(i -> System.out.print(i + " "));
            System.out.println();
            floatArrayList.forEach(f -> System.out.print(f + " "));
            System.out.println();
            stringArrayList.forEach(s -> System.out.print(s + " "));
            System.out.println();

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

class FileProcessor {
    public void parseFile(File file, Set<Integer> integerList, Set<Float> floatList, Set<String> stringList) {
        try {
            List<String> lines = Files.readAllLines(file.toPath());
            for (String line : lines) {
                parseLine(line, integerList, floatList, stringList);
            }
        } catch (IOException e) {
            System.err.println("Ошибка при чтении файла " + file.getName() + ": " + e.getMessage());
        }
    }

    private void parseLine(String line, Set<Integer> integerList, Set<Float> floatList, Set<String> stringList) {
        try {
            integerList.add(Integer.parseInt(line));
        } catch (NumberFormatException e1) {
            try {
                floatList.add(Float.parseFloat(line));
            } catch (NumberFormatException e2) {
                stringList.add(line);
            }
        }
    }
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