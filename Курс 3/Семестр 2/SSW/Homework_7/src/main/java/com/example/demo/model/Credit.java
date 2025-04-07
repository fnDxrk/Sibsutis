package com.example.demo.model;

import jakarta.persistence.DiscriminatorValue;
import jakarta.persistence.Entity;
import java.time.LocalDateTime;

@Entity
@DiscriminatorValue("CREDIT")
public class Credit extends Payment {
    private String number;
    private String type;
    private LocalDateTime expDate;

    public Credit() {}

    public Credit(float amount, String paymentStatus, String number, String type, LocalDateTime expDate) {
        super(amount, paymentStatus);
        this.number = number;
        this.type = type;
        this.expDate = expDate;
    }

    public String getNumber() {
        return number;
    }

    public void setNumber(String number) {
        this.number = number;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public LocalDateTime getExpDate() {
        return expDate;
    }

    public void setExpDate(LocalDateTime expDate) {
        this.expDate = expDate;
    }
}
