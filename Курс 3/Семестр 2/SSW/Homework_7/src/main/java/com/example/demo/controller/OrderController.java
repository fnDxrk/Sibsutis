package com.example.demo.controller;

import com.example.demo.model.Order;
import com.example.demo.service.OrderService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/orders")
@Tag(name = "Order Management", description = "API для управления заказами")
public class OrderController {

    private final OrderService orderService;

    public OrderController(OrderService orderService) {
        this.orderService = orderService;
    }

    @PostMapping
    @Operation(summary = "Создать новый заказ")
    public Order createOrder(@RequestBody Order order) {
        return orderService.createOrder(order);
    }

    @GetMapping
    @Operation(summary = "Получить все заказы")
    public List<Order> getAllOrders() {
        return orderService.getAllOrders();
    }

    @GetMapping("/search")
    @Operation(summary = "Поиск заказов")
    public List<Order> searchOrders(
            @RequestParam(required = false) String customerName,
            @RequestParam(required = false) String city,
            @RequestParam(required = false) String status) {
        return orderService.searchOrders(customerName, city, status);
    }
}