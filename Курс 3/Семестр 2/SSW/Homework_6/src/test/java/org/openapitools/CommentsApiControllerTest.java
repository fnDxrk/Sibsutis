package org.openapitools;

import org.apache.coyote.BadRequestException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.openapitools.api.CommentsApiController;
import org.openapitools.model.Comment;
import org.openapitools.service.CommentService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.context.request.NativeWebRequest;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class CommentsApiControllerTest {

    @Mock
    private CommentService commentService;

    @Mock
    private NativeWebRequest request;

    @InjectMocks
    private CommentsApiController commentsApiController;

    private Comment testComment;

    @BeforeEach
    void setUp() {
        testComment = new Comment();
        testComment.setId(1);
        testComment.setContent("Test Comment");
    }

    @Test
    void commentsGet_ShouldReturnListOfComments() {
        when(commentService.getAllComments()).thenReturn(new ArrayList<>(List.of(testComment)));
        ResponseEntity<List<Comment>> response = commentsApiController.commentsGet();
        Assertions.assertEquals(HttpStatus.OK, response.getStatusCode());
        Assertions.assertFalse(response.getBody().isEmpty());
        Assertions.assertEquals(testComment.getContent(), response.getBody().get(0).getContent());
    }

    @Test
    void commentsGet_ShouldReturnNotFound_WhenNoComments() {
        when(commentService.getAllComments()).thenThrow(new NoSuchElementException());
        ResponseEntity<List<Comment>> response = commentsApiController.commentsGet();
        Assertions.assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
    }

    @Test
    void commentsIdGet_ShouldReturnComment() {
        when(commentService.getCommentById(1)).thenReturn(testComment);
        ResponseEntity<Comment> response = commentsApiController.commentsIdGet(1);
        Assertions.assertEquals(HttpStatus.OK, response.getStatusCode());
        Assertions.assertEquals(testComment.getContent(), response.getBody().getContent());
    }

    @Test
    void commentsIdGet_ShouldReturnNotFound_WhenCommentDoesNotExist() {
        when(commentService.getCommentById(1)).thenThrow(new NoSuchElementException());
        ResponseEntity<Comment> response = commentsApiController.commentsIdGet(1);
        Assertions.assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
    }

    @Test
    void commentsPost_ShouldCreateComment() throws BadRequestException {
        when(commentService.AddComment(any(Comment.class))).thenReturn(testComment);
        ResponseEntity<Comment> response = commentsApiController.commentsPost(testComment);
        Assertions.assertEquals(HttpStatus.CREATED, response.getStatusCode());
        Assertions.assertEquals(testComment.getContent(), response.getBody().getContent());
    }

    @Test
    void commentsIdDelete_ShouldDeleteComment() {
        when(commentService.deleteById(1)).thenReturn(true);
        ResponseEntity<Void> response = commentsApiController.commentsIdDelete(1);
        Assertions.assertEquals(HttpStatus.OK, response.getStatusCode());
    }

    @Test
    void commentsIdDelete_ShouldReturnNotFound_WhenCommentDoesNotExist() {
        when(commentService.deleteById(1)).thenReturn(false);
        ResponseEntity<Void> response = commentsApiController.commentsIdDelete(1);
        Assertions.assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
    }

    @Test
    void commentsIdPut_ShouldUpdateComment() throws BadRequestException {
        when(commentService.UpdateComment(eq(1), any(Comment.class))).thenReturn(testComment);
        ResponseEntity<Comment> response = commentsApiController.commentsIdPut(1, testComment);
        Assertions.assertEquals(HttpStatus.OK, response.getStatusCode());
        Assertions.assertEquals(testComment.getContent(), response.getBody().getContent());
    }

    @Test
    void commentsIdPut_ShouldReturnNotFound_WhenCommentDoesNotExist() throws BadRequestException {
        when(commentService.UpdateComment(eq(1), any(Comment.class))).thenThrow(new NoSuchElementException());
        ResponseEntity<Comment> response = commentsApiController.commentsIdPut(1, testComment);
        Assertions.assertEquals(HttpStatus.NOT_FOUND, response.getStatusCode());
    }
}
