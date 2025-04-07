package com.example.demo.model;

import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;

@Entity
@DiscriminatorValue("CHECK")
public class Check extends Payment {
    private String name;
    private String bankId;

    public Check() {}

    public Check(float amount, String paymentStatus, String name, String bankId) {
        super(amount, paymentStatus);
        this.name = name;
        this.bankId = bankId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getBankId() {
        return bankId;
    }

    public void setBankId(String bankId) {
        this.bankId = bankId;
    }
}
