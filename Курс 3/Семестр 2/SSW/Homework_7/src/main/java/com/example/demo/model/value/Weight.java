package com.example.demo.model.value;

import jakarta.persistence.Embeddable;
import java.math.BigDecimal;

@Embeddable
public class Weight {
    private BigDecimal value;

    public Weight() {
    }

    public Weight(BigDecimal value) {
        this.value = value;
    }

    public BigDecimal getValue() {
        return value;
    }

    public void setValue(BigDecimal value) {
        this.value = value;
    }
}
