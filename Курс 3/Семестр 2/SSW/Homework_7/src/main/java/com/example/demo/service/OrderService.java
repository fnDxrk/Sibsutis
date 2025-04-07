package com.example.demo.service;

import com.example.demo.model.Order;
import com.example.demo.model.Address;
import com.example.demo.repository.OrderRepository;
import jakarta.persistence.criteria.Predicate;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class OrderService {

    private final OrderRepository orderRepository;

    public OrderService(OrderRepository orderRepository) {
        this.orderRepository = orderRepository;
    }

    public Order createOrder(Order order) {
        return orderRepository.save(order);
    }

    public List<Order> getAllOrders() {
        return orderRepository.findAll();
    }

    public List<Order> searchOrders(String customerName, String city, String status) {
        if (customerName == null && city == null && status == null) {
            return getAllOrders();
        }

        return orderRepository.findAll(createSearchSpecification(customerName, city, status));
    }

    private Specification<Order> createSearchSpecification(String customerName, String city, String status) {
        return (root, query, cb) -> {
            List<Predicate> predicates = new ArrayList<>();

            if (customerName != null && !customerName.isEmpty()) {
                predicates.add(cb.like(cb.lower(root.get("customer").get("name")),
                        "%" + customerName.toLowerCase() + "%"));
            }

            if (city != null && !city.isEmpty()) {
                predicates.add(cb.equal(root.get("customer").get("address").get("city"), city));
            }

            if (status != null && !status.isEmpty()) {
                predicates.add(cb.equal(root.get("status"), status));
            }

            return cb.and(predicates.toArray(new Predicate[0]));
        };
    }

    public List<Order> findOrdersByCriteria(String customerName, Address address,
                                            LocalDateTime startDate, LocalDateTime endDate,
                                            String paymentType, String paymentStatus,
                                            String orderStatus) {
        Specification<Order> spec = Specification.where(null);

        if (customerName != null && !customerName.isEmpty()) {
            spec = spec.and((root, query, cb) ->
                    cb.like(cb.lower(root.get("customer").get("name")),
                            "%" + customerName.toLowerCase() + "%"));
        }

        if (address != null) {
            if (address.getCity() != null) {
                spec = spec.and((root, query, cb) ->
                        cb.equal(root.get("customer").get("address").get("city"), address.getCity()));
            }
            if (address.getStreet() != null) {
                spec = spec.and((root, query, cb) ->
                        cb.equal(root.get("customer").get("address").get("street"), address.getStreet()));
            }
        }

        if (startDate != null && endDate != null) {
            spec = spec.and((root, query, cb) ->
                    cb.between(root.get("date"), startDate, endDate));
        }

        if (paymentType != null) {
            spec = spec.and((root, query, cb) ->
                    cb.equal(root.get("payment").type().as(String.class), paymentType));
        }

        if (paymentStatus != null) {
            spec = spec.and((root, query, cb) ->
                    cb.equal(root.get("payment").get("paymentStatus"), paymentStatus));
        }

        if (orderStatus != null) {
            spec = spec.and((root, query, cb) ->
                    cb.equal(root.get("status"), orderStatus));
        }

        return orderRepository.findAll(spec);
    }
}