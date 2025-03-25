package org.openapitools.api;

import jakarta.validation.Valid;
import lombok.AllArgsConstructor;
import org.apache.coyote.BadRequestException;
import org.openapitools.model.Comment;


import org.openapitools.repository.CommentRepository;
import org.openapitools.service.CommentService;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.context.request.NativeWebRequest;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Optional;
import jakarta.annotation.Generated;
import org.springframework.web.server.ResponseStatusException;

@Generated(value = "org.openapitools.codegen.languages.SpringCodegen", date = "2025-03-22T19:40:35.628994242Z[Etc/UTC]", comments = "Generator version: 7.13.0-SNAPSHOT")
@RestController
@AllArgsConstructor
@RequestMapping("${openapi.commentManagement.base-path:/api}")
public class CommentsApiController implements CommentsApi {

    private final NativeWebRequest request;
    private final CommentService commentService;

    @Override
    public Optional<NativeWebRequest> getRequest() {
        return Optional.ofNullable(request);
    }

    /**
     * GET /comments : Получить список всех комментариев
     *
     * @return Список комментариев успешно получен (status code 200)
     * or Комментарии не найдены (status code 404)
     */
    @Override
    public ResponseEntity<List<Comment>> commentsGet() {
        ArrayList<Comment> comments = null;
        try {
            comments = commentService.getAllComments();
        } catch (NoSuchElementException e) {
            System.out.println("No comments found");
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
        return ResponseEntity.ok().body(comments);
    }

    /**
     * DELETE /comments/{id} : Удалить комментарий
     *
     * @param id (required)
     * @return Комментарий успешно удалён (status code 204)
     * or Bad request (status code 400)
     * or Комментарий не найден (status code 404)
     */
    @Override
    public ResponseEntity<Void> commentsIdDelete(@PathVariable Integer id) {
        if (commentService.deleteById(id)){
            return ResponseEntity.ok().build();
        }else{
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    /**
     * GET /comments/{id} : Получить комментарий по идентификатору
     *
     * @param id (required)
     * @return Комментарий успешно получен (status code 200)
     * or Комментарий не найден (status code 404)
     */
    @Override
    public ResponseEntity<Comment> commentsIdGet(@PathVariable Integer id) {
        try {
            Comment targetComment = commentService.getCommentById(id);
            return ResponseEntity.ok().body(targetComment);
        } catch (NoSuchElementException e) {
            System.out.println(e.getMessage());
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
    }

    /**
     * PUT /comments/{id} : Обновить существующий комментарий
     *
     * @param id      (required)
     * @param comment Обновлённый объект комментария (required)
     * @return Комментарий успешно обновлён (status code 200)
     * or Bad request (status code 400)
     * or Комментарий не найден (status code 404)
     */
    @Override
    public ResponseEntity<Comment> commentsIdPut(@PathVariable Integer id, @Valid @RequestBody Comment comment) {
        try {
            Comment updatedComment = commentService.UpdateComment(id, comment);
            return ResponseEntity.ok(updatedComment);
        } catch (BadRequestException e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(null);
        } catch (NoSuchElementException e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.NOT_FOUND).body(null);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }

    /**
     * POST /comments : Создать новый комментарий
     *
     * @param comment Объект комментария для создания (required)
     * @return Комментарий успешно создан (status code 201)
     * or Bad request (status code 400)
     * or Такой комментарий уже существует (status code 404)
     */
    @Override
    public ResponseEntity<Comment> commentsPost(@Valid @RequestBody Comment comment) {
        try {
            Comment newComment = commentService.AddComment(comment);
            return ResponseEntity.status(HttpStatus.CREATED).body(newComment);
        } catch (BadRequestException e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(null);
        } catch (ResponseStatusException e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(e.getStatusCode()).body(null);
        } catch (Exception e) {
            System.out.println(e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }


}
