; 3) Являются ли два множества пересекающимися.

(print "Task 1")

(defun task_1(lst_1 lst_2)
    (cond 
      ((or (null lst_1) (null lst_2)) nil)
      ((member (car lst_1) lst_2) t)
      (t (task_1 (cdr lst_1) lst_2))
    )
)

(print (task_1 '(1 2 3 4 5) '(8 8 8 9 1)))

; 6) Возвращающую разность двух множеств, т.е. множество из элементов первого
; множества, не входящих во второе

(terpri)
(print "Task 2")
(defun task_2(lst_1 lst_2)
  (cond
    ((null lst_1) nil)
    ((member (car lst_1) lst_2) (task_2 (cdr lst_1) lst_2))
    (t (cons (car lst_1) (task_2 (cdr lst_1) lst_2)))
  )
)

(print (task_2 '(0 2 0 4 5) '(8 8 8 9 2)))

; 11) Выполняющий определенную операцию над соответствующими элементами двух
; списков (Используйте применяющий функционал FUNCALL). Проверьте работу
;функционала для операций:
; - выбор максимального элемента (функциональный аргумент – лямбда выражение);
; - деление (функциональный аргумент – имя встроенной функции /).

(terpri)
(print "Task 3")
(defun task_3 (func list1 list2)
  (cond
    ((or (null list1) (null list2)) nil)
    (t (cons (funcall func (car list1) (car list2)) 
             (task_3 func (cdr list1) (cdr list2))))))

(print (task_3 (lambda (x y) (max x y)) '(1 5 3 7) '(1 2 8 6 8)))

(print (task_3 '/ '(10 20 30) '(2 4 5)))


