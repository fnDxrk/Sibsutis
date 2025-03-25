package org.openapitools.service;

import jakarta.transaction.Transactional;
import lombok.AllArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.apache.coyote.BadRequestException;
import org.openapitools.model.Comment;
import org.openapitools.repository.CommentRepository;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import java.util.ArrayList;
import java.util.NoSuchElementException;
import java.util.Optional;

@Service
@Transactional
@RequiredArgsConstructor
public class CommentService {
    private final CommentRepository commentRepository;

    public ArrayList<Comment> getAllComments() {
        ArrayList<Comment> comments = new ArrayList<>();
        try {
            comments.addAll(commentRepository.findAll());
        } catch (Exception e) {
            throw new NoSuchElementException("Comments not found");
        }
        return comments;
    }

    public boolean deleteById(Integer id) {
        if(commentRepository.existsById(id)) {
            commentRepository.deleteById(id);
            return true;
        }else{
            return false;
        }
    }

    public Comment getCommentById(Integer id) {
        if (commentRepository.existsById(id)) {
            return commentRepository.findById(id).get();
        }else {
            throw new NoSuchElementException("Comment with id " + id + " not found");
        }
    }

    public Comment AddComment(Comment newComment) throws BadRequestException, ResponseStatusException {
        if (newComment == null) {
            throw new BadRequestException("Comment is nullable");
        }
        return commentRepository.save(newComment);
    }

    public Comment UpdateComment(Integer id, Comment newComment) throws BadRequestException, NoSuchElementException {
        CheckPutValid(id, newComment);
        return commentRepository.save(newComment);
    }

    private void CheckPutValid(Integer id, Comment newComment) throws BadRequestException, NoSuchElementException {
        if (newComment == null || id == null) {
            throw new BadRequestException("Comment or Comment ID is nullable");
        }

        if (!commentRepository.existsById(id)) {
            throw new NoSuchElementException("Comment with id " + id + " not found");
        }

        Comment existingComment = commentRepository.findById(id).get();
        assert newComment.getAuthorId() != null;
        if (!newComment.getAuthorId().equals(existingComment.getAuthorId())) {
            throw new BadRequestException("Author in comment with id " + id + " is not the same");
        }

        assert newComment.getPostId() != null;
        if (!newComment.getPostId().equals(existingComment.getPostId())) {
            throw new BadRequestException("Post in comment with id " + id + " is not the same");
        }

        newComment.setAuthorId(existingComment.getAuthorId());
        newComment.setPostId(existingComment.getPostId());
        newComment.setId(existingComment.getId());
    }

}
