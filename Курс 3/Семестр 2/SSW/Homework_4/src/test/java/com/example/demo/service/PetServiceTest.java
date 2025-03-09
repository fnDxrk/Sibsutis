package com.example.demo.service;

import com.example.demo.model.Pet;
import com.example.demo.repository.PetRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Optional;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class PetServiceTest {

    @Mock
    private PetRepository petRepository;

    @InjectMocks
    private PetService petService;

    private Pet testPet;

    @BeforeEach
    void setUp() {
        testPet = new Pet();
        testPet.setId(1L);
        testPet.setName("Buddy");
    }

    @Test
    void addPet_ShouldSavePetWithGeneratedId() {
        when(petRepository.save(any(Pet.class))).thenAnswer(invocation -> {
            Pet pet = invocation.getArgument(0);
            pet.setId(1L);
            return pet;
        });

        Pet savedPet = petService.addPet(new Pet());
        assertNotNull(savedPet.getId());
        verify(petRepository).save(any(Pet.class));
    }

    @Test
    void updatePet_WhenPetExists_ShouldUpdatePet() {
        when(petRepository.findById(1L)).thenReturn(Optional.of(testPet));
        when(petRepository.save(any(Pet.class))).thenReturn(testPet);

        Pet updatedPet = petService.updatePet(testPet);
        assertEquals("Buddy", updatedPet.getName());
        verify(petRepository).save(testPet);
    }

    @Test
    void updatePet_WhenPetNotFound_ShouldThrowException() {
        when(petRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(RuntimeException.class, () -> petService.updatePet(testPet));
        verify(petRepository, never()).save(any());
    }

    @Test
    void getPetById_WhenPetExists_ShouldReturnPet() {
        when(petRepository.findById(1L)).thenReturn(Optional.of(testPet));

        Pet foundPet = petService.getPetById(1L);
        assertEquals(1L, foundPet.getId());
    }

    @Test
    void getPetById_WhenPetNotFound_ShouldThrowException() {
        when(petRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(RuntimeException.class, () -> petService.getPetById(1L));
    }

    @Test
    void deletePet_WhenPetExists_ShouldDeletePet() {
        when(petRepository.findById(1L)).thenReturn(Optional.of(testPet));

        petService.deletePet(1L);
        verify(petRepository).deleteById(1L);
    }

    @Test
    void deletePet_WhenPetNotFound_ShouldThrowException() {
        when(petRepository.findById(1L)).thenReturn(Optional.empty());

        assertThrows(RuntimeException.class, () -> petService.deletePet(1L));
        verify(petRepository, never()).deleteById(any());
    }
}