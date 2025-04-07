package com.example.demo.model;

import com.example.demo.model.value.Weight;
import jakarta.persistence.*;

@Entity
@Table(name = "items")
public class Item {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Embedded
    private Weight shippingWeight;

    private String description;

    public Item() {}

    public Item(Weight shippingWeight, String description) {
        this.shippingWeight = shippingWeight;
        this.description = description;
    }

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public Weight getShippingWeight() {
        return shippingWeight;
    }

    public void setShippingWeight(Weight shippingWeight) {
        this.shippingWeight = shippingWeight;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }
}
