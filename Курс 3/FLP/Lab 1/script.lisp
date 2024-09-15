; 1. Напишите сложную функцию, используя композиции функций CAR и CDR, которая
; возвращает атом * при применении к следующему списку:
; 3) ((1 ((*) 2 3)))

(print "Task 1")
; car -> cdr -> car -> car -> car

(defun task_1 (lst)
  (car (car (car (cdr (car lst))))))

(print (task_1 '((1 ((*) 2 3)))))



; 2. Объясните работу функций:
; 3) (cons '(+ 1 2) '(+ 4 6)

(terpri)
(print "Task 2")

(print (cons '(+ 1 2) '(+ 4 6)))

; Функция cons создаёт точечную пару
; В результате у нас получается список,
; головой которого становится первый аргумент -> список (+ 1 2)
; а хвостом становится второй аргумент -> список (+ 4 6)
; И получается -> ((+ 1 2) + 4 6)



; 3. Из атомов 1, 2, 3, nil создайте указанные списки двумя способами:
; а) с помощью композиций функций CONS;
; б) с помощью композиций функций LIST.
; 3) (((1 2 3)))

(terpri)
(print "Task 3")
; cons 3 nil -> cons 2 -> cons 1 -> cons nil -> cons nil
; list -> list -> list 1 2 3  

(print (cons ( cons ( cons 1 (cons 2 (cons 3 nil))) nil) nil))
(print (list (list (list 1 2 3))))



; 4. С помощью DEFUN определите функцию, которая возвращает измененный список по
; заданию (в теле функции разрешается использовать только следующие встроенные
; функции: CAR, CDR, CONS, APPEND, LIST, LAST, BUTLAST с одним аргументом).
; Проверьте её работу, организуя обращение к функции со списками разной длины.
; 3) Функция меняет местами первый и предпоследний элементы списка

(terpri)
(print "Task 4")

(defun task_4 (lst)
  (if (null (cdr lst))
      lst
      (append 
        (last (butlast lst))
        (cdr (butlast (butlast lst)))
        (list (car lst))
        (list (car (last lst))))))

(defparameter list1 '(1))
(defparameter list2 '(1 2 3 4 5))
(defparameter list3 '(1 2 3 4 5 6 7 8 9 10))

(print (task_4 list1))
(print (task_4 list2))
(print (task_4 list3))
