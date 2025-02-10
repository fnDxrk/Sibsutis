import org.example.DataClassifier;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DataClassifierTest {
    private DataClassifier classifier;

    @BeforeEach
    void setUp() {
        classifier = new DataClassifier();
    }

    @Test
    void classify_CorrectlyCategorizesData() {
        classifier.classify("123");
        classifier.classify("45.67");
        classifier.classify("hello");

        assertTrue(classifier.getIntegers().contains(123));
        assertTrue(classifier.getFloats().contains(45.67f));
        assertTrue(classifier.getStrings().contains("hello"));
    }
}