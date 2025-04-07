package com.example.demo.model.value;

import jakarta.persistence.Embeddable;

@Embeddable
public class Measurement {
    private String name;
    private String symbol;

    public Measurement() {}

    public Measurement(String name, String symbol) {
        this.name = name;
        this.symbol = symbol;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }
}
