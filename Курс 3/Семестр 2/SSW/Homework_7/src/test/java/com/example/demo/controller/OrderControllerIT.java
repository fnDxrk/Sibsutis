package com.example.demo.controller;

import com.example.demo.model.Customer;
import com.example.demo.model.Order;
import com.example.demo.model.Cash;
import com.example.demo.model.Address;
import com.example.demo.repository.CustomerRepository;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
@Transactional  // Добавьте эту аннотацию для обеспечения транзакции
class OrderControllerIT {

    private static final MediaType JSON = MediaType.APPLICATION_JSON;

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private CustomerRepository customerRepository;

    @Test
    void shouldCreateOrderViaApi() throws Exception {
        // Создаём и сохраняем Customer в БД
        Customer customer = new Customer("API Customer", new Address("SPb", "Nevsky", "190000"));
        customer = customerRepository.save(customer);  // Сохраняем customer в базе данных

        // Создаём Cash payment
        Cash payment = new Cash(200.0f, "PAID", 200.0f);

        // Создаём Order с уже сохранённым customer
        Order order = new Order();
        order.setDate(LocalDateTime.now());
        order.setStatus("CREATED");
        order.setCustomer(customer);  // Связываем с сохранённым customer
        order.setPayment(payment);

        // Отправляем POST-запрос для создания заказа через API
        mockMvc.perform(post("/api/orders")
                        .contentType(JSON)
                        .content(objectMapper.writeValueAsString(order)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("CREATED"))
                .andExpect(jsonPath("$.customer.name").value("API Customer"))
                .andExpect(jsonPath("$.payment.paymentStatus").value("PAID"));
    }
}
