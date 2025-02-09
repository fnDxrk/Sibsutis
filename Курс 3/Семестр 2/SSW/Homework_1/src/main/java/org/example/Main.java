package org.example;

import java.io.IOException;

public class Main {
    public static void main(String[] args) {
        try {
            Arguments arguments = ArgumentParser.parse(args);
            FileProcessor fileProcessor = new FileProcessor(arguments);
            fileProcessor.processFiles();
        } catch (IllegalArgumentException | IOException e) {
            System.err.println("Ошибка: " + e.getMessage());
        }
    }
}