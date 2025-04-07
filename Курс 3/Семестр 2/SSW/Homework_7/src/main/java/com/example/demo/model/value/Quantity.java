package com.example.demo.model.value;

import jakarta.persistence.Embeddable;

@Embeddable
public class Quantity {
    private int value;

    public Quantity() {}

    public Quantity(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }

    public void setValue(int value) {
        this.value = value;
    }
}
