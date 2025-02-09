package org.example;

import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;

public class DataClassifier {
    private final Set<Integer> integerSet = new HashSet<>();
    private final Set<Double> floatSet = new HashSet<>();
    private final Set<String> stringSet = new HashSet<>();

    public void classify(String line) {
        if (tryParse(line, Integer::parseInt, integerSet)) return;
        if (tryParse(line, Double::parseDouble, floatSet)) return;
        stringSet.add(line);
    }

    private <T> boolean tryParse(String str, Function<String, T> parser, Set<T> set) {
        try {
            set.add(parser.apply(str));
            return true;
        } catch (NumberFormatException e) {
            return false;
        }
    }
}
