package com.example.demo.service;

import com.example.demo.model.Pet;
import com.example.demo.repository.PetRepository;
import org.springframework.stereotype.Service;

@Service
public class PetService {

    private final PetRepository petRepository;

    public PetService(PetRepository petRepository) {
        this.petRepository = petRepository;
    }

    public Pet addPet(Pet pet) {
        return petRepository.save(pet);
    }

    public Pet updatePet(Pet pet) {
        if (pet.getId() == null || petRepository.findById(pet.getId()).isEmpty()) {
            throw new RuntimeException("Pet not found");
        }
        return petRepository.save(pet);
    }

    public Pet getPetById(Long petId) {
        return petRepository.findById(petId)
                .orElseThrow(() -> new RuntimeException("Pet not found"));
    }

    public void deletePet(Long petId) {
        if (petRepository.findById(petId).isEmpty()) {
            throw new RuntimeException("Pet not found");
        }
        petRepository.deleteById(petId);
    }
}
