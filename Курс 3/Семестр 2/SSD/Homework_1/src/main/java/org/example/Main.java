package org.example;

import java.io.File;

public class Main {
    public static void main(String[] args) {
        String outputPath = ".";
        String prefix = "";
        boolean appendEnabled = false;
        StatsMode statsMode = StatsMode.NONE;

        for (int i = 0; i < args.length; i++) {
            switch(args[i]) {
                case "-o" -> {
                    if (i + 1 < args.length) {
                        outputPath = args[++i];
                    } else {
                        System.err.println("Ошибка! Отсутствует путь после параметра -o");
                        System.exit(1);
                    }
                }
                case "-p" -> {
                    if (i + 1 < args.length) {
                        prefix = args[++i];
                    } else {
                        System.err.println("Ошибка! Отсутствует префикс после параметра -p");
                        System.exit(1);
                    }
                }
                case "-a"-> appendEnabled = true;
                case "-s" -> statsMode = StatsMode.SHORT;
                case "-f" -> statsMode = StatsMode.FULL;
                default -> {
                    System.err.println("Ошибка! Указан неверный параметр: " + args[i]);
                    System.exit(1);
                }
            }
        }

        String integersFiles = outputPath + File.separator + prefix + "integers.txt";
        String floatsFiles = outputPath + File.separator + prefix + "floats.txt";
        String stringsFiles = outputPath + File.separator +  prefix + "strings.txt";

        System.out.println("Путь для результатов: " + outputPath);
        System.out.println("Префикс: " + prefix);
        System.out.println("Режим добавления: " + appendEnabled);
        System.out.println("Режим статистики: " + statsMode);
        System.out.println("Полный путь до файла с числами: " + integersFiles);
    }
}

enum StatsMode {
    NONE, SHORT, FULL
}