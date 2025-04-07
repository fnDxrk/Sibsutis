package com.example.demo.model;

import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;

@Entity
@DiscriminatorValue("CASH")
public class Cash extends Payment {
    private float cashTendered;

    public Cash() {}

    public Cash(float amount, String paymentStatus, float cashTendered) {
        super(amount, paymentStatus);
        this.cashTendered = cashTendered;
    }

    public float getCashTendered() {
        return cashTendered;
    }

    public void setCashTendered(float cashTendered) {
        this.cashTendered = cashTendered;
    }
}
