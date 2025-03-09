package com.example.demo.controller;

import com.example.demo.model.Category;
import com.example.demo.model.Pet;
import com.example.demo.model.Tag;
import com.example.demo.service.PetService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.setup.MockMvcBuilders;

import java.util.Collections;

import static org.hamcrest.Matchers.is;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

class PetControllerTest {

    private MockMvc mockMvc;
    private PetService petService;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        petService = mock(PetService.class);
        PetController petController = new PetController(petService);
        mockMvc = MockMvcBuilders.standaloneSetup(petController).build();
        objectMapper = new ObjectMapper();
    }

    @Test
    void addPet_ShouldReturnCreatedPet() throws Exception {
        // Arrange
        Pet inputPet = new Pet(null, "doggie", null, null, "available");
        Pet createdPet = new Pet(1L, "doggie", null, null, "available");

        when(petService.addPet(any(Pet.class))).thenReturn(createdPet);

        // Act & Assert
        mockMvc.perform(post("/api/v3/pet")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(inputPet)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id", is(1)))
                .andExpect(jsonPath("$.name", is("doggie")))
                .andExpect(jsonPath("$.status", is("available")));

        verify(petService, times(1)).addPet(any(Pet.class));
    }

    @Test
    void updatePet_ShouldReturnUpdatedPet() throws Exception {
        // Arrange
        Pet inputPet = new Pet(1L, "doggie-updated", null, null, "pending");

        when(petService.updatePet(any(Pet.class))).thenReturn(inputPet);

        // Act & Assert
        mockMvc.perform(put("/api/v3/pet")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(inputPet)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id", is(1)))
                .andExpect(jsonPath("$.name", is("doggie-updated")))
                .andExpect(jsonPath("$.status", is("pending")));

        verify(petService, times(1)).updatePet(any(Pet.class));
    }

    @Test
    void updatePet_WhenNotFound_ShouldReturnNotFound() throws Exception {
        // Arrange
        Pet inputPet = new Pet(1L, "doggie-updated", null, null, "pending");

        when(petService.updatePet(any(Pet.class)))
                .thenThrow(new RuntimeException("Pet not found"));

        // Act & Assert
        mockMvc.perform(put("/api/v3/pet")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(inputPet)))
                .andExpect(status().isNotFound())
                .andExpect(content().string("Pet not found"));

        verify(petService, times(1)).updatePet(any(Pet.class));
    }

    @Test
    void getPetById_ShouldReturnPet() throws Exception {
        // Arrange
        Pet pet = new Pet(1L, "doggie",
                new Category(1L, "Dogs"),
                Collections.singletonList(new Tag(1L, "cute")),
                "available");

        when(petService.getPetById(1L)).thenReturn(pet);

        // Act & Assert
        mockMvc.perform(get("/api/v3/pet/1")
                        .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.id", is(1)))
                .andExpect(jsonPath("$.name", is("doggie")))
                .andExpect(jsonPath("$.category.name", is("Dogs")))
                .andExpect(jsonPath("$.tags[0].name", is("cute")))
                .andExpect(jsonPath("$.status", is("available")));

        verify(petService, times(1)).getPetById(1L);
    }

    @Test
    void getPetById_WhenNotFound_ShouldReturnNotFound() throws Exception {
        // Arrange
        when(petService.getPetById(1L))
                .thenThrow(new RuntimeException("Pet not found"));

        // Act & Assert
        mockMvc.perform(get("/api/v3/pet/1")
                        .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isNotFound())
                .andExpect(content().string("Pet not found"));

        verify(petService, times(1)).getPetById(1L);
    }

    @Test
    void deletePet_ShouldReturnOk() throws Exception {
        // Arrange
        doNothing().when(petService).deletePet(1L);

        // Act & Assert
        mockMvc.perform(delete("/api/v3/pet/1"))
                .andExpect(status().isOk());

        verify(petService, times(1)).deletePet(1L);
    }

    @Test
    void deletePet_WhenError_ShouldReturnBadRequest() throws Exception {
        // Arrange
        doThrow(new RuntimeException("Invalid pet ID")).when(petService).deletePet(1L);

        // Act & Assert
        mockMvc.perform(delete("/api/v3/pet/1"))
                .andExpect(status().isBadRequest())
                .andExpect(content().string("Invalid pet ID"));

        verify(petService, times(1)).deletePet(1L);
    }
}